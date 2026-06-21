#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab Docker 镜像构建脚本
# 支持交互式 + 非交互式两种模式
#
# 交互式: ./build.sh
# 非交互: ./build.sh collect
#          ./build.sh all
#          ./build.sh --mirror cn collect
#          ./build.sh --rebuild-base collect
#          ./build.sh --no-rebuild-base infer
#          ./build.sh --base                # 只构建基础镜像
#
# 注: 训练在云端进行, 不需要本地训练镜像
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
# 构建上下文根: unitree/  (gr00t_mjlab_autodl 的父目录)
# 必须包含兄弟仓库: unitree_rl_mjlab/, gr00t_mjlab_autodl/
UNITREE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_PREFIX="gr00t-mjlab"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
MIRROR=""
REBUILD_BASE=false
NO_REBUILD_BASE=false
BASE_ONLY=false

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_banner() { echo -e "${BOLD}${BLUE}$1${NC}"; }
log_info()   { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $1"; }

prompt_select() {
    local prompt="$1"; shift; local options=("$@")
    echo "" >&2
    echo -e "${BOLD}${prompt}${NC}" >&2
    for i in "${!options[@]}"; do
        echo " $((i+1))) ${options[$i]}" >&2
    done
    read -p "请选择 [1-${#options[@]}]: " choice
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#options[@]}" ]; then
        log_error "无效选择" >&2
        prompt_select "$prompt" "${options[@]}"
        return
    fi
    echo "$((choice-1))"
}

check_docker() {
    if ! command -v docker &>/dev/null; then log_error "未找到 docker"; exit 1; fi
    if ! docker info &>/dev/null; then log_error "Docker 守护进程未运行"; exit 1; fi
}

# ── 构建基础镜像 ─────────────────────────────────────────────────────────
build_base() {
    local dockerfile="$DOCKER_DIR/Dockerfile.base"
    local tag="${IMAGE_PREFIX}-base:latest"

    log_info "构建基础镜像..."
    log_info " Dockerfile: $dockerfile  标签: $tag"
    [ -n "$MIRROR" ] && log_info " 镜像源: 国内 ($MIRROR)" || log_info " 镜像源: 官方"

    local t0=$(date +%s)
    if docker build --file "$dockerfile" --tag "$tag" \
        --build-arg BUILD_DATE="$BUILD_DATE" --build-arg VERSION="latest" \
        --build-arg MIRROR="$MIRROR" "$UNITREE_ROOT"; then
        local t1=$(date +%s)
        local sz=$(docker images "$tag" --format "{{.Size}}" | head -1)
        log_info "✅ 基础镜像构建成功: $tag (${sz}, $((t1-t0))s)"
    else
        local t1=$(date +%s)
        log_error "❌ 基础镜像构建失败 (耗时 $((t1-t0))s)"
        exit 1
    fi
}

# ── 检查基础镜像是否需要重建 ────────────────────────────────────────────
needs_base_build() {
    if [ "$REBUILD_BASE" = true ]; then return 0; fi
    if ! docker image inspect "${IMAGE_PREFIX}-base:latest" &>/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ── 构建变体镜像 ─────────────────────────────────────────────────────────
build_variant() {
    local variant="$1"
    local dockerfile="$DOCKER_DIR/Dockerfile.$variant"
    local tag="${IMAGE_PREFIX}-${variant}:latest"
    local base_tag="${IMAGE_PREFIX}-base:latest"

    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile 不存在: $dockerfile"
        return 1
    fi

    # 确保基础镜像存在
    if [ "$NO_REBUILD_BASE" = true ]; then
        if ! docker image inspect "$base_tag" &>/dev/null 2>&1; then
            log_info "基础镜像不存在, 先构建..."
            build_base
        else
            log_info "跳过基础镜像重建: $base_tag"
        fi
    elif needs_base_build; then
        build_base
    else
        log_info "基础镜像已是最新: $base_tag"
    fi

    log_info "构建 ${variant} 镜像..."
    log_info " Dockerfile: $dockerfile  标签: $tag"
    local t0=$(date +%s)
    if docker build --file "$dockerfile" --tag "$tag" \
        --build-arg BUILD_DATE="$BUILD_DATE" --build-arg VERSION="latest" \
        --build-arg MIRROR="$MIRROR" "$UNITREE_ROOT"; then
        local t1=$(date +%s)
        local sz=$(docker images "$tag" --format "{{.Size}}" | head -1)
        log_info "✅ ${variant} 镜像构建成功: $tag (${sz}, $((t1-t0))s)"
    else
        local t1=$(date +%s)
        log_error "❌ ${variant} 镜像构建失败 (耗时 $((t1-t0))s)"
        return 1
    fi
}

# ── 显示镜像列表 ─────────────────────────────────────────────────────────
show_images() {
    echo ""
    log_banner "── 当前镜像列表 ──"
    echo ""
    local images
    images=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -E "^${IMAGE_PREFIX}" || true)
    if [ -n "$images" ]; then
        echo "$images"
    else
        log_warn "暂无 ${IMAGE_PREFIX} 镜像"
    fi
    echo ""
}

# ── 交互式主流程 ─────────────────────────────────────────────────────────
run_interactive() {
    log_banner "═══════════════════════════════════════════════════════════════"
    log_banner "        gr00t-mjlab Docker 镜像构建"
    log_banner "═══════════════════════════════════════════════════════════════"
    echo ""
    check_docker
    show_images

    local idx=$(prompt_select "请选择要构建的镜像:" \
        "collect   (数据采集, 最小, ~5GB)" \
        "infer     (推理验证, 中等, ~7GB)" \
        "all       (构建 collect + infer)")

    local VARIANTS=()
    case "$idx" in
        0) VARIANTS=(collect) ;;
        1) VARIANTS=(infer) ;;
        2) VARIANTS=(collect infer) ;;
    esac

    local action_idx=$(prompt_select "请选择操作类型:" \
        "完全重建 (基础镜像 + 变体)" \
        "只更新变体 (跳过基础镜像)" \
        "只构建基础镜像")

    echo ""
    log_banner "── 构建确认 ──"
    echo "  镜像: ${VARIANTS[*]}"
    echo ""

    case "$action_idx" in
        0) REBUILD_BASE=true; for v in "${VARIANTS[@]}"; do build_variant "$v" || exit 1; done ;;
        1) NO_REBUILD_BASE=true; for v in "${VARIANTS[@]}"; do build_variant "$v" || exit 1; done ;;
        2) BASE_ONLY=true; build_base ;;
    esac

    show_images
}

# ── 非交互式主流程 ───────────────────────────────────────────────────────
run_noninteractive() {
    check_docker
    local VARIANTS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --mirror)         shift; MIRROR="${1:-cn}" ;;
            --rebuild-base)   REBUILD_BASE=true ;;
            --no-rebuild-base) NO_REBUILD_BASE=true ;;
            --base)           BASE_ONLY=true ;;
            collect|infer) VARIANTS+=("$1") ;;
            all)              VARIANTS=(collect infer); break ;;
            *) log_error "未知参数: $1"; exit 1 ;;
        esac
        shift
    done

    if [ "$BASE_ONLY" = true ]; then
        build_base
        show_images
        exit 0
    fi

    if [ ${#VARIANTS[@]} -eq 0 ]; then
        log_error "请指定要构建的镜像: collect|infer|all"
        exit 1
    fi

    for v in "${VARIANTS[@]}"; do
        build_variant "$v" || exit 1
    done
    show_images
}

# ── 入口 ─────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    run_interactive
else
    run_noninteractive "$@"
fi
