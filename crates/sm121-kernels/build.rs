// crates/sm121-kernels/build.rs

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ptx_root = manifest_dir.parent().unwrap().parent().unwrap().join("ptx");

    // Watch the ptx directory tree so new files trigger rebuild
    println!("cargo:rerun-if-changed={}", ptx_root.display());
    watch_dirs(&ptx_root);

    let ptxas = find_ptxas();
    let cpp = find_cpp();
    let ptx_files = discover_ptx_files(&ptx_root);

    // (stem, cubin_path, preprocessed_ptx_path). The cubin is the sm_121a fast path;
    // the preprocessed PTX is embedded too so the driver can JIT it on other SM12x archs
    // (forward-compat fat artifact) while keeping the zero-runtime-toolkit property.
    let mut kernels: Vec<(String, PathBuf, PathBuf)> = Vec::new();

    for ptx_file in &ptx_files {
        println!("cargo:rerun-if-changed={}", ptx_file.display());

        let stem = ptx_file.file_stem().unwrap().to_str().unwrap().to_string();

        // Detect per-kernel arch flag from PTX source.
        // A line like `// BUILD_ARCH: sm_121f` in the first 10 lines overrides the default sm_121a.
        // Required for MXFP4 / NVFP4 block-scaled kernels that need family-mode compilation.
        let arch_flag = detect_arch(ptx_file);

        // Step 1: Preprocess (resolve #include directives)
        let preprocessed_path = out_dir.join(format!("{stem}.preprocessed.ptx"));
        let cpp_output = Command::new(&cpp)
            .args([
                "-P",
                "-I",
                ptx_root.join("common").to_str().unwrap(),
                "-I",
                ptx_file.parent().unwrap().to_str().unwrap(),
                ptx_file.to_str().unwrap(),
                "-o",
                preprocessed_path.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to run cpp: {e}"));

        if !cpp_output.status.success() {
            let stderr = String::from_utf8_lossy(&cpp_output.stderr);
            panic!(
                "cpp preprocessing failed for {}:\n{}",
                ptx_file.display(),
                stderr
            );
        }

        // Step 2: Assemble to cubin
        let cubin_path = out_dir.join(format!("{stem}.cubin"));
        let ptxas_output = Command::new(&ptxas)
            .args([
                &format!("-arch={arch_flag}"),
                "-O3",
                "--warn-on-spills",
                "-o",
                cubin_path.to_str().unwrap(),
                preprocessed_path.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to run ptxas: {e}"));

        if !ptxas_output.status.success() {
            let stderr = String::from_utf8_lossy(&ptxas_output.stderr);
            panic!("ptxas failed for {}:\n{}", ptx_file.display(), stderr);
        }

        // Print any warnings from ptxas
        let stderr = String::from_utf8_lossy(&ptxas_output.stderr);
        if !stderr.is_empty() {
            eprintln!("cargo:warning=ptxas {}: {}", stem, stderr.trim());
        }

        kernels.push((stem, cubin_path, preprocessed_path));
    }

    // Expand the codegen templates. Each ptx/attention/templates/<family>.variants manifest
    // lists `<stem> <entry> [SPARK_* flags...]` rows; expand each row from <family>.ptx.in via
    // cpp -DSPARK_ENTRY=<entry> -D<flag>... then assemble it like any other kernel. The
    // templates are the source of truth for the collapsed FA families; the hand-PTX lives in
    // archive/ as the identity reference only (scripts/check_codegen_identity.sh proves the
    // expansion is byte-identical to it).
    let templates_dir = ptx_root.join("attention").join("templates");
    let common_inc = ptx_root.join("common");
    for (template, stem, entry, flags) in discover_template_variants(&templates_dir) {
        println!("cargo:rerun-if-changed={}", template.display());
        let arch_flag = detect_arch(&template);

        let preprocessed_path = out_dir.join(format!("{stem}.preprocessed.ptx"));
        let mut cpp_args: Vec<String> = vec![
            "-P".into(),
            "-I".into(),
            common_inc.to_str().unwrap().into(),
            "-I".into(),
            templates_dir.to_str().unwrap().into(),
            format!("-DSPARK_ENTRY={entry}"),
        ];
        for flag in &flags {
            cpp_args.push(format!("-D{flag}"));
        }
        cpp_args.push(template.to_str().unwrap().into());
        cpp_args.push("-o".into());
        cpp_args.push(preprocessed_path.to_str().unwrap().into());

        let cpp_output = Command::new(&cpp)
            .args(&cpp_args)
            .output()
            .unwrap_or_else(|e| panic!("failed to run cpp: {e}"));
        if !cpp_output.status.success() {
            panic!(
                "cpp failed expanding template variant {stem}:\n{}",
                String::from_utf8_lossy(&cpp_output.stderr)
            );
        }

        let cubin_path = out_dir.join(format!("{stem}.cubin"));
        let ptxas_output = Command::new(&ptxas)
            .args([
                &format!("-arch={arch_flag}"),
                "-O3",
                "--warn-on-spills",
                "-o",
                cubin_path.to_str().unwrap(),
                preprocessed_path.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to run ptxas: {e}"));
        if !ptxas_output.status.success() {
            panic!(
                "ptxas failed for template variant {stem}:\n{}",
                String::from_utf8_lossy(&ptxas_output.stderr)
            );
        }
        let stderr = String::from_utf8_lossy(&ptxas_output.stderr);
        if !stderr.is_empty() {
            eprintln!("cargo:warning=ptxas {}: {}", stem, stderr.trim());
        }

        kernels.push((stem, cubin_path, preprocessed_path));
    }

    // Also watch all .ptxh header files for changes
    if let Ok(entries) = fs::read_dir(ptx_root.join("common")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "ptxh") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    // Generate embedded_kernels.rs
    let gen_path = out_dir.join("embedded_kernels.rs");
    let mut f = fs::File::create(&gen_path).unwrap();

    writeln!(f, "// Auto-generated by build.rs. Do not edit.").unwrap();
    writeln!(f, "use std::sync::LazyLock;").unwrap();
    writeln!(f).unwrap();
    writeln!(
        f,
        "pub static KERNEL_REGISTRY: LazyLock<HashMap<&'static str, &'static [u8]>> = LazyLock::new(|| {{"
    )
    .unwrap();
    writeln!(f, "    let mut m = HashMap::new();").unwrap();

    for (name, cubin_path, _) in &kernels {
        writeln!(
            f,
            "    m.insert(\"{name}\", include_bytes!(\"{}\").as_slice());",
            cubin_path.display()
        )
        .unwrap();
    }

    writeln!(f, "    m").unwrap();
    writeln!(f, "}});").unwrap();

    // Forward-compat: embed the preprocessed PTX per kernel so the loader can JIT
    // it on any SM12x arch when the sm_121a cubin is rejected. cpp/ifdef/include are
    // already resolved, so this PTX is exactly what produced the cubin.
    writeln!(f).unwrap();
    writeln!(
        f,
        "pub static KERNEL_PTX_REGISTRY: LazyLock<HashMap<&'static str, &'static [u8]>> = LazyLock::new(|| {{"
    )
    .unwrap();
    writeln!(f, "    let mut m = HashMap::new();").unwrap();
    for (name, _, ptx_path) in &kernels {
        writeln!(
            f,
            "    m.insert(\"{name}\", include_bytes!(\"{}\").as_slice());",
            ptx_path.display()
        )
        .unwrap();
    }
    writeln!(f, "    m").unwrap();
    writeln!(f, "}});").unwrap();
}

fn find_ptxas() -> PathBuf {
    if let Ok(p) = env::var("PTXAS") {
        return PathBuf::from(p);
    }
    if let Ok(cuda) = env::var("CUDA_PATH") {
        let p = PathBuf::from(cuda).join("bin/ptxas");
        if p.exists() {
            return p;
        }
    }
    for path in [
        "/usr/local/cuda/bin/ptxas",
        "/opt/cuda/bin/ptxas",
        "/usr/bin/ptxas",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }
    eprintln!("cargo:warning=ptxas not found in standard paths. Set PTXAS=/path/to/ptxas or install CUDA Toolkit.");
    PathBuf::from("ptxas")
}

fn find_cpp() -> PathBuf {
    if let Ok(p) = env::var("CPP") {
        return PathBuf::from(p);
    }
    for path in ["/usr/bin/cpp", "/usr/local/bin/cpp"] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("cpp")
}

fn watch_dirs(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            watch_dirs(&path);
        }
    }
}

fn discover_ptx_files(root: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    visit_dir(root, &mut results);
    results.sort();
    results
}

/// Parse every `<family>.variants` manifest in `templates_dir` into
/// `(template_path, file_stem, entry_name, flags)` rows. Manifest line format is
/// `<stem> <entry> [SPARK_* flags... | -]`; lines starting with `#` are comments.
fn discover_template_variants(templates_dir: &Path) -> Vec<(PathBuf, String, String, Vec<String>)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(templates_dir) else {
        return out;
    };
    let mut manifests: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "variants"))
        .collect();
    manifests.sort();
    for manifest in manifests {
        let family = manifest.file_stem().unwrap().to_str().unwrap();
        let template = templates_dir.join(format!("{family}.ptx.in"));
        let Ok(content) = fs::read_to_string(&manifest) else {
            continue;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(stem), Some(entry)) = (parts.next(), parts.next()) else {
                continue;
            };
            let flags: Vec<String> = parts.filter(|f| *f != "-").map(|f| f.to_string()).collect();
            out.push((template.clone(), stem.to_string(), entry.to_string(), flags));
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

fn detect_arch(ptx_file: &Path) -> String {
    // Default SM121a (architecture-accelerated). Override with `// BUILD_ARCH: sm_121f`
    // (Family mode) for kernels that use mma.kind::mxf4/nvf4.block_scale instructions.
    let Ok(content) = fs::read_to_string(ptx_file) else {
        return "sm_121a".to_string();
    };
    for line in content.lines().take(10) {
        if let Some(rest) = line.trim_start().strip_prefix("// BUILD_ARCH:") {
            let arch = rest.trim();
            if !arch.is_empty() {
                return arch.to_string();
            }
        }
    }
    "sm_121a".to_string()
}

fn visit_dir(dir: &Path, results: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Skip non-build dirs:
            //   templates/ holds the .ptx.in codegen sources (expanded below, not assembled bare)
            //   archive/   holds the frozen hand-PTX kept only as the codegen identity reference
            //   generated/ is the standalone-script output for the CI identity gate (the build
            //              expands templates straight to OUT_DIR, so assembling these would duplicate)
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "templates" | "archive" | "generated") {
                continue;
            }
            visit_dir(&path, results);
        } else if path.extension().is_some_and(|e| e == "ptx") {
            // Only include files with .entry directives (actual kernels, not headers)
            if let Ok(content) = fs::read_to_string(&path) {
                if content.contains(".entry") {
                    results.push(path);
                }
            }
        }
    }
}
