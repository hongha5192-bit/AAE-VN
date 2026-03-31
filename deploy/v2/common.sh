#!/bin/bash
set -euo pipefail

v2_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

v2_repo_root() {
    cd "$(v2_script_dir)/../.." && pwd
}

v2_default_work_dir() {
    if [ -n "${WORK:-}" ]; then
        printf '%s\n' "$WORK"
        return
    fi

    if [ -n "${KAGGLE_WORKING_DIR:-}" ]; then
        if [ -d "$KAGGLE_WORKING_DIR/aae_v2" ]; then
            printf '%s\n' "$KAGGLE_WORKING_DIR/aae_v2"
            return
        fi
        if [ -d "$KAGGLE_WORKING_DIR/aae-v2" ]; then
            printf '%s\n' "$KAGGLE_WORKING_DIR/aae-v2"
            return
        fi
        printf '%s\n' "$KAGGLE_WORKING_DIR/aae_v2"
        return
    fi

    if [ -d /kaggle/working ]; then
        if [ -d /kaggle/working/aae_v2 ]; then
            printf '%s\n' "/kaggle/working/aae_v2"
            return
        fi
        if [ -d /kaggle/working/aae-v2 ]; then
            printf '%s\n' "/kaggle/working/aae-v2"
            return
        fi
        printf '%s\n' "/kaggle/working/aae_v2"
        return
    fi

    if [ -d /workspace ]; then
        printf '%s\n' "/workspace/v2"
        return
    fi

    printf '%s\n' "$(v2_repo_root)/.runtime/v2"
}

v2_activate_env() {
    local work_dir="${1:-$(v2_default_work_dir)}"
    local env_name="${ENV_NAME:-verl041}"

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
        if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            source "$conda_base/etc/profile.d/conda.sh"
            if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
                conda activate "$env_name"
                return 0
            fi
        fi
    fi

    local micromamba_bin="$work_dir/micromamba/bin/micromamba"
    local mamba_root_prefix="$work_dir/micromamba/root-prefix"
    if [ -x "$micromamba_bin" ] && [ -d "$mamba_root_prefix/envs/$env_name" ]; then
        export MAMBA_ROOT_PREFIX="$mamba_root_prefix"
        eval "$("$micromamba_bin" shell hook -s bash -r "$MAMBA_ROOT_PREFIX")"
        micromamba activate "$env_name"
        return 0
    fi

    local venv_path="$work_dir/venv/$env_name"
    # Kaggle should use micromamba path from setup.sh.
    # Keep venv fallback only when explicitly enabled.
    if [ "${ALLOW_VENV_FALLBACK:-0}" = "1" ] && [ -f "$venv_path/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$venv_path/bin/activate"
        return 0
    fi

    echo "ERROR: Could not activate environment '$env_name'." >&2
    echo "  Expected conda env '$env_name' or micromamba env at $mamba_root_prefix/envs/$env_name" >&2
    if [ "${ALLOW_VENV_FALLBACK:-0}" != "1" ]; then
        echo "  Note: venv fallback is disabled by default. Set ALLOW_VENV_FALLBACK=1 to enable it." >&2
    fi
    return 1
}
