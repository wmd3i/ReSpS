#!/bin/bash

# EAGLE Action Analysis Script - Clean Output Version
# Analyzes evaluation logs to extract unique actions by step

# Function to check if output is to terminal
is_terminal() {
    [ -t 1 ]
}

# Function to print with optional colors
print_colored() {
    local color_code="$1"
    local message="$2"
    if is_terminal; then
        echo -e "\033[${color_code}m${message}\033[0m"
    else
        echo "$message"
    fi
}

print_header() {
    print_colored "0;36" "$1"
}

print_section() {
    print_colored "1;33" "$1"
}

print_info() {
    print_colored "0;32" "$1"
}

print_warning() {
    print_colored "0;31" "$1"
}

# Help function
show_help() {
    cat << EOF
EAGLE Action Analysis Script

USAGE:
    $0 [TARGET] [OPTIONS]

ARGUMENTS:
    TARGET          Directory or log file to analyze (default: current directory)

OPTIONS:
    -h, --help      Show this help message
    -d, --detailed  Show detailed step-by-step breakdown and actual actions
    -b, --benchmark Filter by benchmark name (e.g., alpaca, mt_bench)
    -s, --summary   Show only summary (no per-file details)
    -v, --verbose   Show verbose debugging information

EXAMPLES:
    # Analyze all logs in current directory
    $0

    # Analyze specific experiment with detailed output
    $0 -d 20250727_132601_optimized_ppo_cu64

    # Show only summary, no colors (good for file output)
    $0 -s > analysis_results.txt

    # Analyze specific benchmark with verbose output
    $0 -v -b alpaca
EOF
}

# Parse command line arguments
TARGET="."
DETAILED=false
BENCHMARK=""
SUMMARY_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--detailed)
            DETAILED=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        -s|--summary)
            SUMMARY_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Function to extract and analyze actions from a log file
analyze_log_file() {
    local log_file="$1"
    local temp_dir=$(mktemp -d)
    local actions_file="$temp_dir/actions.txt"
    local unique_actions_file="$temp_dir/unique_actions.txt"
    local step_analysis_file="$temp_dir/step_analysis.txt"
    
    if [ "$VERBOSE" = true ]; then
        echo "DEBUG: Analyzing file: $log_file"
        echo "DEBUG: Temp directory: $temp_dir"
    fi
    
    # Extract action lines with more flexible pattern matching
    grep -E "(Action \(tokens=.*\) first appeared at step|Action.*tokens=.*depth=.*top_k=.*step)" "$log_file" > "$actions_file" 2>/dev/null
    
    if [ ! -s "$actions_file" ]; then
        if [ "$VERBOSE" = true ]; then
            echo "DEBUG: No action lines found in $log_file"
            # Try to show what patterns exist in the file
            echo "DEBUG: Looking for any 'Action' mentions:"
            grep -i "action" "$log_file" | head -5 2>/dev/null || echo "DEBUG: No 'action' mentions found"
        fi
        echo "0 0 NO_ACTIONS"
        rm -rf "$temp_dir"
        return
    fi
    
    if [ "$VERBOSE" = true ]; then
        echo "DEBUG: Found $(wc -l < "$actions_file") action lines"
        echo "DEBUG: First few action lines:"
        head -3 "$actions_file"
    fi
    
    # Extract unique actions
    sed -E 's/.*Action \(([^)]*)\).*/\1/' "$actions_file" | sort -u > "$unique_actions_file"
    
    # Extract step information
    sed -E 's/.*step ([0-9]+).*/\1/' "$actions_file" | sort -u -n > "$step_analysis_file"
    
    local unique_actions=$(wc -l < "$unique_actions_file")
    local unique_steps=$(wc -l < "$step_analysis_file")
    
    echo "$unique_actions $unique_steps $temp_dir"
}

# Function to show detailed actions
show_detailed_actions() {
    local log_file="$1"
    local temp_dir="$2"
    
    if [ ! -d "$temp_dir" ]; then
        return
    fi
    
    local actions_file="$temp_dir/actions.txt"
    local unique_actions_file="$temp_dir/unique_actions.txt"
    
    if [ -f "$unique_actions_file" ] && [ -s "$unique_actions_file" ]; then
        echo ""
        print_section "    All unique actions found:"
        while read -r action; do
            echo "      - Action ($action)"
        done < "$unique_actions_file"
        
        if [ "$DETAILED" = true ]; then
            echo ""
            print_section "    Step-by-step breakdown:"
            
            # Create step-by-step analysis
            sed -E 's/.*Action \(([^)]*)\).*step ([0-9]+).*/\2:\1/' "$actions_file" | sort -n | \
            while IFS=':' read -r step action; do
                echo "      Step $step: Action ($action)"
            done
        fi
    fi
}

# Main analysis
print_header "========================================================================"
print_header "EAGLE Action Analysis - Unique Actions by Step"
print_header "========================================================================"

# Find log files
if [ -f "$TARGET" ]; then
    LOG_FILES=("$TARGET")
else
    # Find evaluation log files with better pattern matching
    mapfile -t LOG_FILES < <(find "$TARGET" -type f \( -name "*_evaluation.log" -o -name "*evaluation*.log" -o -name "*.log" \) | grep -E "(evaluation|eval)" | sort)
fi

# Filter by benchmark if specified
if [ -n "$BENCHMARK" ]; then
    mapfile -t FILTERED_FILES < <(printf '%s\n' "${LOG_FILES[@]}" | grep -i "$BENCHMARK")
    LOG_FILES=("${FILTERED_FILES[@]}")
fi

if [ ${#LOG_FILES[@]} -eq 0 ]; then
    print_warning "No evaluation log files found in $TARGET"
    if [ -n "$BENCHMARK" ]; then
        print_warning "Try removing the benchmark filter (-b $BENCHMARK)"
    fi
    echo ""
    echo "Searched for files matching patterns:"
    echo "  - *_evaluation.log"
    echo "  - *evaluation*.log" 
    echo "  - *.log (in directories containing 'evaluation' or 'eval')"
    exit 1
fi

print_info "Found ${#LOG_FILES[@]} log file(s) to analyze"
echo ""

if [ "$VERBOSE" = true ]; then
    echo "Files to analyze:"
    printf '%s\n' "${LOG_FILES[@]}"
    echo ""
fi

# Global summary variables
declare -A ALL_UNIQUE_ACTIONS
TOTAL_POLICIES=0
TOTAL_ACTIONS=0
POLICIES_WITH_ACTIONS=0

# Analyze each log file
for log_file in "${LOG_FILES[@]}"; do
    # Extract policy and benchmark name
    policy_name=$(basename "$(dirname "$(dirname "$log_file")")")
    benchmark_name=$(basename "$log_file" | sed 's/_evaluation\.log$//' | sed 's/\.log$//')
    
    if [ "$SUMMARY_ONLY" = false ]; then
        echo "----------------------------------------"
        print_section "Analyzing: $log_file"
    fi
    
    # Analyze the file
    read -r unique_actions unique_steps temp_dir < <(analyze_log_file "$log_file")
    
    if [ "$unique_actions" -eq 0 ] || [ "$temp_dir" = "NO_ACTIONS" ]; then
        if [ "$SUMMARY_ONLY" = false ]; then
            print_warning "  No actions found in this log file"
            if [ "$VERBOSE" = true ]; then
                echo "  File size: $(wc -c < "$log_file" 2>/dev/null || echo "unknown") bytes"
                echo "  File lines: $(wc -l < "$log_file" 2>/dev/null || echo "unknown") lines"
            fi
        fi
        continue
    fi
    
    TOTAL_POLICIES=$((TOTAL_POLICIES + 1))
    TOTAL_ACTIONS=$((TOTAL_ACTIONS + unique_actions))
    POLICIES_WITH_ACTIONS=$((POLICIES_WITH_ACTIONS + 1))
    
    if [ "$SUMMARY_ONLY" = false ]; then
        print_info "  Policy: $policy_name"
        print_info "  Benchmark: $benchmark_name"
        print_info "  Unique actions: $unique_actions"
        print_info "  Steps with actions: $unique_steps"
        
        # Show detailed actions if requested
        if [ "$DETAILED" = true ] || [ "$VERBOSE" = true ]; then
            show_detailed_actions "$log_file" "$temp_dir"
        fi
    fi
    
    # Collect unique actions for global summary
    if [ -d "$temp_dir" ] && [ -f "$temp_dir/unique_actions.txt" ]; then
        while read -r action; do
            ALL_UNIQUE_ACTIONS["$action"]=1
        done < "$temp_dir/unique_actions.txt"
        rm -rf "$temp_dir"
    fi
done

# Print global summary
echo ""
print_header "========================================================================"
print_header "GLOBAL SUMMARY"
print_header "========================================================================"

print_info "Total log files analyzed: $TOTAL_POLICIES"
print_info "Log files with actions: $POLICIES_WITH_ACTIONS"
print_info "Total unique actions across all policies: ${#ALL_UNIQUE_ACTIONS[@]}"

if [ ${#ALL_UNIQUE_ACTIONS[@]} -gt 0 ]; then
    echo ""
    print_section "All unique actions found (sorted by tokens, depth, top_k):"
    
    # Sort actions by tokens, depth, top_k
    for action in "${!ALL_UNIQUE_ACTIONS[@]}"; do
        echo "$action"
    done | sort -t'=' -k2,2n -k4,4n -k6,6n | while read -r action; do
        echo "  - Action ($action)"
    done
    
    echo ""
    print_section "Action distribution analysis:"
    
    # Analyze token distribution
    echo "Token values found:"
    for action in "${!ALL_UNIQUE_ACTIONS[@]}"; do
        echo "$action" | sed -E 's/tokens=([0-9]+).*/\1/'
    done | sort -n | uniq -c | while read -r count tokens; do
        echo "  tokens=$tokens: $count unique combinations"
    done
    
    echo ""
    echo "Depth values found:"
    for action in "${!ALL_UNIQUE_ACTIONS[@]}"; do
        echo "$action" | sed -E 's/.*depth=([0-9]+).*/\1/'
    done | sort -n | uniq -c | while read -r count depth; do
        echo "  depth=$depth: $count unique combinations"
    done
    
    echo ""
    echo "Top_k values found:"
    for action in "${!ALL_UNIQUE_ACTIONS[@]}"; do
        echo "$action" | sed -E 's/.*top_k=([0-9]+).*/\1/'
    done | sort -n | uniq -c | while read -r count top_k; do
        echo "  top_k=$top_k: $count unique combinations"
    done
fi

echo ""
print_header "========================================================================"
print_header "Analysis Complete"
print_header "========================================================================"

if [ $POLICIES_WITH_ACTIONS -eq 0 ]; then
    echo ""
    print_warning "No actions were found in any log files!"
    echo "This might indicate:"
    echo "  1. The log files don't contain action information"
    echo "  2. The action format has changed"
    echo "  3. The evaluation hasn't been run with action logging enabled"
    echo ""
    echo "Try running with -v (verbose) flag to see more debugging information."
fi
