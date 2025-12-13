#!/bin/bash

# Multi-GPU Script Generator for EAGLE RL Training
# This script generates multiple GPU-specific training scripts based on user preferences
#
# Features:
#   - Pre-built combinations: 8 standard parameter combinations
#   - Custom parameters: Define your own parameter combinations for each script
#   - Multi-GPU support with round-robin or custom GPU assignment
#   - Flexible date/time options and file handling
#
# Usage:
#   bash generate_multi_gpu_scripts.sh                    # Interactive mode
#   bash generate_multi_gpu_scripts.sh 1 2 3 4 5 6 7 8   # All pre-built combinations
#   bash generate_multi_gpu_scripts.sh 1 3 5 7            # Specific pre-built combinations
#   bash generate_multi_gpu_scripts.sh --datetime YYYYMMDD_HHMMSS 1 2 3 4  # With custom date/time
#   bash generate_multi_gpu_scripts.sh --overwrite 1 2 3 4  # Overwrite existing scripts

echo "=== EAGLE RL Multi-GPU Script Generator ==="
echo ""

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage examples:"
echo "  bash $0                    # Interactive mode (choose pre-built or custom combinations)"
echo "  bash $0 1 2 3 4 5 6 7 8   # Pre-built combinations (non-interactive)"
echo "  bash $0 1 3 5 7            # Specific pre-built combinations"
echo "  bash $0 --datetime 20250207_151418_cu18 1 2 3 4  # With custom date/time"
echo "  bash $0 --overwrite 1 2 3 4  # Overwrite existing scripts (keep same date/time)"
echo "  bash $0 --datetime hello_world 1 2 3 4  # With custom folder name"
    echo ""
    echo "Combination modes (interactive):"
echo "  1. Pre-built combinations - 8 standard parameter combinations"
echo "  2. Custom parameters - define your own parameter combinations for each script"
    echo ""
    echo "Custom parameter format (for mode 2):"
echo "  6 space-separated values (0 or 1): 'RUN_STANDARD_VERSION RUN_OFL_VERSION RUN_STANDARD RUN_CONTEXT_ONLY RUN_MAX_ENTROPY RUN_NO_MAX_ENTROPY'"
echo "  Example: '1 0 1 0 1 0' = Standard Version + Standard State + Max Entropy"
    echo ""
    echo "Date/Time options (interactive mode):"
echo "  1. Use current date/time (default)"
echo "  2. Specify custom date/time (any string allowed)"
echo "  3. Use simple folder name (no date/time)"
echo "  4. Overwrite existing scripts (keep same date/time)"
echo "  5. Delete old files and start fresh"
echo "  6. Custom folder name (enter your own name)"
    echo ""
    echo "File handling options (when existing scripts found):"
    echo "  1. Keep old files (default) - new scripts alongside existing ones"
    echo "  2. Delete old files - remove existing scripts and generate new ones"
    echo "  3. Overwrite old files - replace existing scripts with new ones"
    echo ""
fi

# Parse command line arguments for custom date/time and overwrite
CUSTOM_DATETIME_ARG=""
OVERWRITE_MODE=false
if [ $# -gt 0 ] && [ "$1" == "--datetime" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: --datetime requires a date/time argument"
        exit 1
    fi
    CUSTOM_DATETIME_ARG="$2"
    
    echo "Using command-line specified date/time: $CUSTOM_DATETIME_ARG"
    # Remove the --datetime and its argument from the command line
    shift 2
elif [ $# -gt 0 ] && [ "$1" == "--overwrite" ]; then
    OVERWRITE_MODE=true
    echo "Overwrite mode enabled - will use existing folder with same date/time"
    # Remove the --overwrite argument from the command line
    shift 1
fi

# Set the base script name (fixed)
BASE_SCRIPT="test_optimized_ppo_modes_comparison_single"
if [ ! -f "${BASE_SCRIPT}.sh" ]; then
    echo "Error: ${BASE_SCRIPT}.sh not found!"
    exit 1
fi
echo "Using base script: ${BASE_SCRIPT}.sh"

# Model Selection
echo ""
echo "=== Model Selection ==="
echo "Available models:"
echo "1. LLaMA3.1-8B (default) - meta-llama/Llama-3.1-8B-Instruct"
echo "2. Vicuna-13B - lmsys/vicuna-13b-v1.3"
echo "3. LLaMA3.3-70B - meta-llama/Llama-3.3-70B-Instruct"
echo ""

# Check for command line model selection
MODEL_CHOICE=""
if [ $# -gt 0 ]; then
    # Non-interactive mode - use default model unless specified
    MODEL_CHOICE=1
    echo "Non-interactive mode: Using default LLaMA3.1-8B model"
else
    # Interactive mode - ask user
    read -p "Choose model (1/2/3, default: 1): " MODEL_CHOICE
    if [ -z "$MODEL_CHOICE" ]; then
        MODEL_CHOICE=1
        echo "Using default: LLaMA3.1-8B"
    fi
fi

# Set model paths based on selection
case $MODEL_CHOICE in
    1)
        MODEL_NAME="LLaMA3.1-8B"
        MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
        GEN_SCRIPT="gen_ea_answer_llama3chat_rl"
        BASELINE_SCRIPT="gen_baseline_answer_llama3chat"
        EAGLE3_SCRIPT="gen_ea_answer_llama3chat"
        ;;
    2)
        MODEL_NAME="Vicuna-13B"
        MODEL_PATH="yuhuili/EAGLE3-Vicuna1.3-13B"
        BASE_MODEL_PATH="lmsys/vicuna-13b-v1.3"
        GEN_SCRIPT="gen_ea_answer_vicuna_rl"
        BASELINE_SCRIPT="gen_baseline_answer_vicuna"
        EAGLE3_SCRIPT="gen_ea_answer_vicuna"
        ;;
    3)
        MODEL_NAME="LLaMA3.3-70B"
        MODEL_PATH="yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
        BASE_MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"
        GEN_SCRIPT="gen_ea_answer_llama3chat_rl"
        BASELINE_SCRIPT="gen_baseline_answer_llama3chat"
        EAGLE3_SCRIPT="gen_ea_answer_llama3chat"
        ;;
    *)
        echo "Invalid model choice. Using default LLaMA3.1-8B"
        MODEL_NAME="LLaMA3.1-8B"
        MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
        GEN_SCRIPT="gen_ea_answer_llama3chat_rl"
        BASELINE_SCRIPT="gen_baseline_answer_llama3chat"
        EAGLE3_SCRIPT="gen_ea_answer_llama3chat"
        ;;
esac

echo "Selected model: $MODEL_NAME"
echo "  EAGLE model path: $MODEL_PATH"
echo "  Base model path: $BASE_MODEL_PATH"
echo "  Generation script: $GEN_SCRIPT"
echo "  Baseline script: $BASELINE_SCRIPT"
echo ""

# Create model short name for filenames and directories
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')

# Model name inclusion option
echo ""
echo "=== Model Name Inclusion ==="
echo "Do you want to include the model name in generated file and folder names?"
echo "1. Yes - Include model name (e.g., test_op__vicuna13b_multi_gpu_20250729_184423/) [default]"
echo "2. No - Use legacy naming (e.g., test_op__multi_gpu_20250729_184423/)"
echo ""

INCLUDE_MODEL_NAME=""
if [ $# -gt 0 ]; then
    # Non-interactive mode - use new naming with model name by default
    INCLUDE_MODEL_NAME=1
    echo "Non-interactive mode: Using new naming with model name"
else
    # Interactive mode - ask user
    read -p "Choose naming style (1/2, default: 1): " INCLUDE_MODEL_NAME
    if [ -z "$INCLUDE_MODEL_NAME" ]; then
        INCLUDE_MODEL_NAME=1
        echo "Using default: Include model name in generated names"
    fi
fi

case $INCLUDE_MODEL_NAME in
    1)
        echo "Will include model name ($MODEL_SHORT_NAME) in generated names"
        USE_MODEL_IN_NAMES=true
        ;;
    2)
        echo "Using legacy naming without model name"
        USE_MODEL_IN_NAMES=false
        ;;
    *)
        echo "Invalid choice. Using default: Include model name in generated names"
        USE_MODEL_IN_NAMES=true
        ;;
esac

# Create a shorter prefix for generated files (first 15 characters)
if [ ${#BASE_SCRIPT} -gt 7 ]; then
    FILE_PREFIX="${BASE_SCRIPT:0:7}"
    echo "Using shortened prefix for files: $FILE_PREFIX"
else
    FILE_PREFIX="$BASE_SCRIPT"
fi

# Get number of available GPUs
read -p "Enter number of available GPUs (default: 4): " NUM_GPUS
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=4
    echo "Using default: $NUM_GPUS GPUs"
fi

# Function to get combination parameters
get_combination_params() {
    local combo=$1
    if [ "$COMBINATION_MODE" -eq 1 ]; then
        # Pre-built combinations mode
        case $combo in
            1) echo "1 0 1 0 1 0" ;;  # Standard Version + Standard State + Max Entropy
            2) echo "1 0 1 0 0 1" ;;  # Standard Version + Standard State + No Max Entropy
            3) echo "1 0 0 1 1 0" ;;  # Standard Version + Context Only + Max Entropy
            4) echo "1 0 0 1 0 1" ;;  # Standard Version + Context Only + No Max Entropy
            5) echo "0 1 1 0 1 0" ;;  # OFL Version + Standard State + Max Entropy
            6) echo "0 1 1 0 0 1" ;;  # OFL Version + Standard State + No Max Entropy
            7) echo "0 1 0 1 1 0" ;;  # OFL Version + Context Only + Max Entropy
            8) echo "0 1 0 1 0 1" ;;  # OFL Version + Context Only + No Max Entropy
        esac
    else
        # Custom parameters mode
        local index=$((combo - 1))
        echo "${CUSTOM_PARAMETERS[$index]}"
    fi
}

# Function to get combination description
get_combination_desc() {
    local combo=$1
    if [ "$COMBINATION_MODE" -eq 1 ]; then
        # Pre-built combinations mode
        case $combo in
            1) echo "standard_hiddstat_maxent" ;;
            2) echo "standard_hiddstat_noent" ;;
            3) echo "standard_context_maxent" ;;
            4) echo "standard_context_noent" ;;
            5) echo "ofl_hiddstat_maxent" ;;
            6) echo "ofl_hiddstat_noent" ;;
            7) echo "ofl_context_maxent" ;;
            8) echo "ofl_context_noent" ;;
        esac
    else
        # Custom parameters mode
        local index=$((combo - 1))
        echo "${CUSTOM_DESCRIPTIONS[$index]}"
    fi
}

# GPU Assignment Configuration (will be configured after combinations are selected)
GPU_ASSIGNMENT_METHOD=1  # Default to round-robin
declare -a GPU_ASSIGNMENTS

# Define all possible combinations
declare -a COMBINATIONS=()
declare -a CUSTOM_PARAMETERS=()  # For storing custom parameter combinations
declare -a CUSTOM_DESCRIPTIONS=()  # For storing custom descriptions

# Check if command line arguments are provided (force pre-built mode)
if [ $# -gt 0 ]; then
    COMBINATION_MODE=1
    echo "Command line arguments detected. Using pre-built combinations mode."
else
    echo ""
    echo "=== Combination Mode Selection ==="
    echo "Choose how to define parameter combinations:"
    echo "1. Use pre-built combinations (default) - 8 standard combinations"
    echo "2. Define custom parameters for each script"
    echo ""
    read -p "Choose mode (1/2, default: 1): " COMBINATION_MODE

    if [ -z "$COMBINATION_MODE" ]; then
        COMBINATION_MODE=1
        echo "Using default: Pre-built combinations"
    fi
fi

if [ "$COMBINATION_MODE" -eq 1 ]; then
    # Pre-built combinations mode
    echo ""
    echo "Available pre-built combinations:"
    echo "1. Standard Version + Standard State + Max Entropy"
    echo "2. Standard Version + Standard State + No Max Entropy"
    echo "3. Standard Version + Context Only + Max Entropy"
    echo "4. Standard Version + Context Only + No Max Entropy"
    echo "5. OFL Version + Standard State + Max Entropy"
    echo "6. OFL Version + Standard State + No Max Entropy"
    echo "7. OFL Version + Context Only + Max Entropy"
    echo "8. OFL Version + Context Only + No Max Entropy"
    echo ""

    # Check for command line arguments
    if [ $# -gt 0 ]; then
        SELECTED_COMBINATIONS="$*"
        echo "Using command line arguments: $SELECTED_COMBINATIONS"
    else
        # Get user selection
        echo ""
        echo "Enter combination numbers to run:"
        echo "  - Enter 'all' for all 8 combinations"
        echo "  - Enter specific numbers (space-separated, e.g., 1 3 5 7)"
        echo "  - Enter '1 2 3 4 5 6 7 8' for all combinations"
        echo "  - Press Enter for default (all 8 combinations)"
        read -p "Your choice: " SELECTED_COMBINATIONS
        
        # Set default if empty
        if [ -z "$SELECTED_COMBINATIONS" ]; then
            SELECTED_COMBINATIONS="1 2 3 4 5 6 7 8"
            echo "Using default: $SELECTED_COMBINATIONS"
        fi
    fi
elif [ "$COMBINATION_MODE" -eq 2 ]; then
    # Custom parameters mode
    echo ""
    echo "=== Custom Parameters Mode ==="
    echo "You will define custom parameter combinations for each script."
    echo ""
    echo "Parameter format: 6 space-separated values (0 or 1)"
    echo "Position 1: RUN_STANDARD_VERSION (1=yes, 0=no)"
    echo "Position 2: RUN_OFL_VERSION (1=yes, 0=no)"
    echo "Position 3: RUN_STANDARD (1=yes, 0=no) - standard state"
    echo "Position 4: RUN_CONTEXT_ONLY (1=yes, 0=no) - context only state"
    echo "Position 5: RUN_MAX_ENTROPY (1=yes, 0=no) - max entropy PPO"
    echo "Position 6: RUN_NO_MAX_ENTROPY (1=yes, 0=no) - no max entropy PPO"
    echo ""
    echo "Examples:"
    echo "  '1 0 1 0 1 0' = Standard Version + Standard State + Max Entropy"
    echo "  '0 1 1 0 0 1' = OFL Version + Standard State + No Max Entropy"
    echo "  '1 0 0 1 1 1' = Standard Version + Context Only + Both Entropy modes"
    echo ""
    
    read -p "How many custom scripts do you want to generate? " NUM_CUSTOM_SCRIPTS
    
    if ! [[ "$NUM_CUSTOM_SCRIPTS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Invalid number of scripts. Please enter a positive integer."
        exit 1
    fi
    
    # Collect custom parameters for each script
    for ((i=1; i<=NUM_CUSTOM_SCRIPTS; i++)); do
        echo ""
        echo "Script $i/$NUM_CUSTOM_SCRIPTS:"
        read -p "Enter parameters (6 values, e.g., '1 0 1 0 1 0'): " CUSTOM_PARAMS
        
        # Validate input
        read -r p1 p2 p3 p4 p5 p6 <<< "$CUSTOM_PARAMS"
        if [[ ! "$p1" =~ ^[01]$ ]] || [[ ! "$p2" =~ ^[01]$ ]] || [[ ! "$p3" =~ ^[01]$ ]] || \
           [[ ! "$p4" =~ ^[01]$ ]] || [[ ! "$p5" =~ ^[01]$ ]] || [[ ! "$p6" =~ ^[01]$ ]]; then
            echo "Error: Invalid parameters. Each value must be 0 or 1."
            echo "You entered: '$CUSTOM_PARAMS'"
            exit 1
        fi
        
        # Basic validation: at least one version should be selected
        if [ "$p1" -eq 0 ] && [ "$p2" -eq 0 ]; then
            echo "Error: At least one version (Standard or OFL) must be selected."
            exit 1
        fi
        
        # Basic validation: at least one state type should be selected
        if [ "$p3" -eq 0 ] && [ "$p4" -eq 0 ]; then
            echo "Error: At least one state type (Standard or Context Only) must be selected."
            exit 1
        fi
        
        # Basic validation: at least one entropy mode should be selected
        if [ "$p5" -eq 0 ] && [ "$p6" -eq 0 ]; then
            echo "Error: At least one entropy mode (Max Entropy or No Max Entropy) must be selected."
            exit 1
        fi
        
        # Store parameters and create description
        CUSTOM_PARAMETERS+=("$CUSTOM_PARAMS")
        
        # Generate description
        VERSION_DESC=""
        if [ "$p1" -eq 1 ] && [ "$p2" -eq 1 ]; then
            VERSION_DESC="both"
        elif [ "$p1" -eq 1 ]; then
            VERSION_DESC="standard"
        elif [ "$p2" -eq 1 ]; then
            VERSION_DESC="ofl"
        fi
        
        STATE_DESC=""
        if [ "$p3" -eq 1 ] && [ "$p4" -eq 1 ]; then
            STATE_DESC="allstat"
        elif [ "$p3" -eq 1 ]; then
            STATE_DESC="hiddstat"
        elif [ "$p4" -eq 1 ]; then
            STATE_DESC="context"
        fi
        
        ENTROPY_DESC=""
        if [ "$p5" -eq 1 ] && [ "$p6" -eq 1 ]; then
            ENTROPY_DESC="allent"
        elif [ "$p5" -eq 1 ]; then
            ENTROPY_DESC="maxent"
        elif [ "$p6" -eq 1 ]; then
            ENTROPY_DESC="noent"
        fi
        
        FULL_DESC="${VERSION_DESC}_${STATE_DESC}_${ENTROPY_DESC}"
        CUSTOM_DESCRIPTIONS+=("$FULL_DESC")
        
        echo "  Parameters: $CUSTOM_PARAMS"
        echo "  Description: $FULL_DESC"
    done
    
    # Set COMBINATIONS array for custom mode (using indices)
    for ((i=1; i<=NUM_CUSTOM_SCRIPTS; i++)); do
        COMBINATIONS+=($i)
    done
    
    echo ""
    echo "Custom parameter combinations defined:"
    for i in "${!CUSTOM_PARAMETERS[@]}"; do
        echo "  Script $((i+1)): ${CUSTOM_PARAMETERS[$i]} (${CUSTOM_DESCRIPTIONS[$i]})"
    done
    
    SELECTED_COMBINATIONS="${COMBINATIONS[*]}"
else
    echo "Invalid combination mode. Using default pre-built combinations."
    COMBINATION_MODE=1
fi

# Parse selected combinations (only for pre-built mode)
if [ "$COMBINATION_MODE" -eq 1 ]; then
    if [[ "$SELECTED_COMBINATIONS" == "all" ]]; then
        COMBINATIONS=(1 2 3 4 5 6 7 8)
    else
        for combo in $SELECTED_COMBINATIONS; do
            if [ "$combo" -ge 1 ] && [ "$combo" -le 8 ]; then
                COMBINATIONS+=($combo)
            else
                echo "Warning: Invalid combination number $combo, skipping..."
            fi
        done
    fi
fi

if [ ${#COMBINATIONS[@]} -eq 0 ]; then
    echo "Error: No valid combinations selected!"
    exit 1
fi

echo ""
if [ "$COMBINATION_MODE" -eq 1 ]; then
    echo "Selected pre-built combinations: ${COMBINATIONS[*]}"
else
    echo "Using custom parameter combinations for ${#COMBINATIONS[@]} scripts"
fi
echo "Number of scripts to generate: ${#COMBINATIONS[@]}"
echo "Available GPUs: $NUM_GPUS"

# Determine which policy versions are needed based on selected combinations
NEED_STANDARD_VERSION=false
NEED_OFL_VERSION=false

if [ "$COMBINATION_MODE" -eq 1 ]; then
    # Pre-built combinations mode
    for combo in "${COMBINATIONS[@]}"; do
        if [ "$combo" -ge 1 ] && [ "$combo" -le 4 ]; then
            NEED_STANDARD_VERSION=true
        elif [ "$combo" -ge 5 ] && [ "$combo" -le 8 ]; then
            NEED_OFL_VERSION=true
        fi
    done
else
    # Custom parameters mode - check custom parameters
    for params in "${CUSTOM_PARAMETERS[@]}"; do
        read -r p1 p2 p3 p4 p5 p6 <<< "$params"
        if [ "$p1" -eq 1 ]; then
            NEED_STANDARD_VERSION=true
        fi
        if [ "$p2" -eq 1 ]; then
            NEED_OFL_VERSION=true
        fi
    done
fi

# Get network architecture configurations (only for needed versions)
echo ""
echo "=== Network Architecture Configuration ==="
echo ""

# Standard version network architecture (only if needed)
if [ "$NEED_STANDARD_VERSION" = true ]; then
    echo "Standard Version Network Architecture:"
    echo "  - Format: comma-separated integers (e.g., '64,64' or '512,256,128')"
    echo "  - Default: '64,64'"
    read -p "Enter network architecture for standard version (default: 64,64): " STANDARD_NET_ARCH
    if [ -z "$STANDARD_NET_ARCH" ]; then
        STANDARD_NET_ARCH="64,64"
        echo "Using default: $STANDARD_NET_ARCH"
    fi
    echo ""
else
    STANDARD_NET_ARCH="64,64"  # Default value (won't be used)
fi

# OFL version network architecture (only if needed)
if [ "$NEED_OFL_VERSION" = true ]; then
    echo "OFL Version Network Architecture:"
    echo "  - Format: '64,64' for same pi/vf or '64,64;128,128' for different pi/vf"
    echo "  - Default: '64,64;64,64'"
    read -p "Enter network architecture for OFL version (default: 64,64;64,64): " OFL_NET_ARCH
    if [ -z "$OFL_NET_ARCH" ]; then
        OFL_NET_ARCH="64,64;64,64"
        echo "Using default: $OFL_NET_ARCH"
    fi
    echo ""
else
    OFL_NET_ARCH="64,64;64,64"  # Default value (won't be used)
fi

# Show summary only for the versions that will be used
echo "Network Architecture Summary:"
if [ "$NEED_STANDARD_VERSION" = true ]; then
    echo "  - Standard Version: $STANDARD_NET_ARCH"
fi
if [ "$NEED_OFL_VERSION" = true ]; then
    echo "  - OFL Version: $OFL_NET_ARCH"
fi
echo ""

# GPU Assignment Configuration
echo ""
echo "=== GPU Assignment Configuration ==="
echo "Choose GPU assignment method:"
echo "1. Automatic round-robin assignment (default) - scripts distributed evenly across GPUs"
echo "2. Custom GPU assignment - specify exact GPU assignments for each script"
read -p "Choose assignment method (1/2, default: 1): " GPU_ASSIGNMENT_METHOD

if [ -z "$GPU_ASSIGNMENT_METHOD" ]; then
    GPU_ASSIGNMENT_METHOD=1
    echo "Using default: Automatic round-robin assignment"
fi

if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    echo ""
    echo "Custom GPU Assignment Mode"
    echo "=========================="
    echo "Enter GPU assignments for each script in the format: gpu1,gpu2;gpu3,gpu4;gpu5;gpu6,gpu7,gpu8"
    echo "Examples:"
    echo "  - '0,1;2,3;4,5;6,7,8,9' for 4 scripts with different GPU allocations"
    echo "  - '0;1;2;3' for 4 scripts, each using 1 GPU"
    echo "  - '0,1,2;3,4,5;6,7;8,9' for 4 scripts with varying GPU counts"
    echo "  - '-1;0;1;2' for 4 scripts, first script uses no GPU (CPU only)"
    echo ""
    echo "Rules:"
    echo "  - Use semicolon (;) to separate different scripts"
    echo "  - Use comma (,) to separate multiple GPUs for one script"
    echo "  - Use -1 to indicate no GPU (CPU only)"
    echo "  - GPU numbers should be 0-based and less than $NUM_GPUS"
    echo "  - You must provide exactly ${#COMBINATIONS[@]} assignments"
    echo ""
    
    read -p "Enter GPU assignments: " CUSTOM_GPU_ASSIGNMENT
    
    # Parse the custom GPU assignment
    IFS=';' read -ra SCRIPT_ASSIGNMENTS <<< "$CUSTOM_GPU_ASSIGNMENT"
    
    if [ ${#SCRIPT_ASSIGNMENTS[@]} -ne ${#COMBINATIONS[@]} ]; then
        echo "Error: Expected ${#COMBINATIONS[@]} GPU assignments, but got ${#SCRIPT_ASSIGNMENTS[@]}"
        exit 1
    fi
    
    # Validate each assignment
    for i in "${!SCRIPT_ASSIGNMENTS[@]}"; do
        assignment="${SCRIPT_ASSIGNMENTS[$i]}"
        IFS=',' read -ra GPUS <<< "$assignment"
        
        for gpu in "${GPUS[@]}"; do
            if [ "$gpu" != "-1" ]; then
                if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
                    echo "Error: Invalid GPU number '$gpu' in assignment $((i+1))"
                    exit 1
                fi
                if [ "$gpu" -ge "$NUM_GPUS" ]; then
                    echo "Error: GPU $gpu is out of range (0-$((NUM_GPUS-1))) in assignment $((i+1))"
                    exit 1
                fi
            fi
        done
        
        # Store the assignment
        GPU_ASSIGNMENTS[$i]="$assignment"
    done
    
    echo ""
    echo "Custom GPU assignments validated:"
    for i in "${!COMBINATIONS[@]}"; do
        combo=${COMBINATIONS[$i]}
        desc=$(get_combination_desc $combo)
        echo "  Script $((i+1)) (Combination $combo - $desc): GPUs [${GPU_ASSIGNMENTS[$i]}]"
    done
else
    echo "Using automatic round-robin assignment"
    # Initialize with round-robin assignment (will be set later)
    for i in "${!COMBINATIONS[@]}"; do
        GPU_ASSIGNMENTS[$i]=""
    done
fi

# Handle date/time selection
if [ "$OVERWRITE_MODE" == true ]; then
    # Overwrite mode - find the most recent generated folder
    echo ""
    echo "Looking for existing generated folders..."
    
    # Find all existing multi-gpu folders for this model or all folders
    if [ "$USE_MODEL_IN_NAMES" = true ]; then
        EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu* 2>/dev/null | sort -r))
    else
        EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null | sort -r))
    fi
    
    if [ ${#EXISTING_FOLDERS[@]} -eq 0 ]; then
        echo "No existing generated folders found. Using current date/time instead."
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATE}"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATE}"
        fi
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
    else
        # Use the most recent folder (first in sorted list)
        MOST_RECENT_FOLDER="${EXISTING_FOLDERS[0]}"
        echo "Found existing folder: $MOST_RECENT_FOLDER"
        
        # Extract date/time from folder name
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_(.+)$ ]]; then
                EXTRACTED_DATE="${BASH_REMATCH[1]}"
                CUSTOM_DATE="$EXTRACTED_DATE"
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing date/time: $CUSTOM_DATE"
                echo "Folder name: $OUTPUT_DIR"
            else
                # Simple folder without date/time
                CUSTOM_DATE=""
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing simple folder: $OUTPUT_DIR"
            fi
        else
            if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__multi_gpu_(.+)$ ]]; then
                EXTRACTED_DATE="${BASH_REMATCH[1]}"
                CUSTOM_DATE="$EXTRACTED_DATE"
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing date/time: $CUSTOM_DATE"
                echo "Folder name: $OUTPUT_DIR"
            else
                # Simple folder without date/time
                CUSTOM_DATE=""
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing simple folder: $OUTPUT_DIR"
            fi
        fi
        echo "This will overwrite existing scripts in this folder."
    fi
elif [ -n "$CUSTOM_DATETIME_ARG" ]; then
    # Use command-line specified date/time
    CUSTOM_DATE="$CUSTOM_DATETIME_ARG"
    if [ "$USE_MODEL_IN_NAMES" = true ]; then
        OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATETIME_ARG}"
    else
        OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATETIME_ARG}"
    fi
    echo "Using command-line specified date/time: $CUSTOM_DATETIME_ARG"
    echo "Folder name: $OUTPUT_DIR"
else
    # Ask about custom date/time
echo ""
echo "Date/Time options:"
echo "1. Use current date/time (default)"
echo "2. Specify custom date/time"
echo "3. Use simple folder name (no date/time)"
echo "4. Overwrite existing scripts (keep same date/time)"
echo "5. Delete old files and start fresh"
echo "6. Custom folder name (enter your own name)"
read -p "Choose option (1/2/3/4/5/6, default: 1): " DATE_OPTION

    if [[ "$DATE_OPTION" == "2" ]]; then
        echo ""
        echo "Enter custom date/time (can be any string)"
        echo "Examples:"
        echo "  - 20241201_143052 (standard format)"
        echo "  - 20250207_151418_cu18 (with suffix)"
        echo "  - my_experiment_v1 (custom name)"
        read -p "Custom date/time: " CUSTOM_DATETIME
        
        CUSTOM_DATE="$CUSTOM_DATETIME"
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATETIME}"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATETIME}"
        fi
        echo "Using custom date/time: $CUSTOM_DATETIME"
        echo "Folder name: $OUTPUT_DIR"
    elif [[ "$DATE_OPTION" == "3" ]]; then
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu"
        fi
        CUSTOM_DATE=""
        echo "Using simple folder name: $OUTPUT_DIR"
    elif [[ "$DATE_OPTION" == "4" ]]; then
        # Overwrite existing scripts - find the most recent generated folder
        echo ""
        echo "Looking for existing generated folders..."
        
        # Find all existing multi-gpu folders for this model or all folders
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu* 2>/dev/null | sort -r))
        else
            EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null | sort -r))
        fi
        
        if [ ${#EXISTING_FOLDERS[@]} -eq 0 ]; then
            echo "No existing generated folders found. Using current date/time instead."
            CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
            if [ "$USE_MODEL_IN_NAMES" = true ]; then
                OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATE}"
            else
                OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATE}"
            fi
            echo "Using current date/time: $CUSTOM_DATE"
            echo "Folder name: $OUTPUT_DIR"
        else
            # Use the most recent folder
            MOST_RECENT_FOLDER="${EXISTING_FOLDERS[0]}"
            echo "Found existing folder: $MOST_RECENT_FOLDER"
            
            # Extract date/time from folder name
            if [ "$USE_MODEL_IN_NAMES" = true ]; then
                if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_(.+)$ ]]; then
                    EXTRACTED_DATE="${BASH_REMATCH[1]}"
                    CUSTOM_DATE="$EXTRACTED_DATE"
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing date/time: $CUSTOM_DATE"
                    echo "Folder name: $OUTPUT_DIR"
                else
                    # Simple folder without date/time
                    CUSTOM_DATE=""
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing simple folder: $OUTPUT_DIR"
                fi
            else
                if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__multi_gpu_(.+)$ ]]; then
                    EXTRACTED_DATE="${BASH_REMATCH[1]}"
                    CUSTOM_DATE="$EXTRACTED_DATE"
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing date/time: $CUSTOM_DATE"
                    echo "Folder name: $OUTPUT_DIR"
                else
                    # Simple folder without date/time
                    CUSTOM_DATE=""
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing simple folder: $OUTPUT_DIR"
                fi
            fi
            echo "This will overwrite existing scripts in this folder."
            
            # Ask for confirmation (default: y since user already chose to overwrite)
            read -p "Do you want to overwrite scripts in $OUTPUT_DIR? (y/n, default: y): " CONFIRM_OVERWRITE
            if [[ "$CONFIRM_OVERWRITE" =~ ^[Nn]$ ]]; then
                echo "Operation cancelled."
                exit 0
            fi
            # Default to yes if no input provided
            if [ -z "$CONFIRM_OVERWRITE" ]; then
                CONFIRM_OVERWRITE="y"
            fi
        fi
    elif [[ "$DATE_OPTION" == "5" ]]; then
        # Delete old files and start fresh
        echo ""
        echo "Delete mode: Looking for existing files to clean up..."
        
        # Delete existing scripts in current directory
        EXISTING_SCRIPTS_CURRENT=$(ls ${FILE_PREFIX}__gpu*_*.sh 2>/dev/null | wc -l)
        if [ $EXISTING_SCRIPTS_CURRENT -gt 0 ]; then
            echo "Found $EXISTING_SCRIPTS_CURRENT existing scripts in current directory."
            rm -f ${FILE_PREFIX}__gpu*_*.sh
            rm -f run_all_scripts.sh launch.sh script_summary.txt
            echo "Deleted existing scripts in current directory."
        fi
        
        # Delete existing multi-gpu folders for this model or all folders
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu* 2>/dev/null))
        else
            EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null))
        fi
        if [ ${#EXISTING_FOLDERS[@]} -gt 0 ]; then
            echo "Found ${#EXISTING_FOLDERS[@]} existing multi-gpu folders:"
            for folder in "${EXISTING_FOLDERS[@]}"; do
                echo "  - $folder"
            done
            read -p "Do you want to delete all existing multi-gpu folders? (y/n, default: n): " DELETE_FOLDERS
            if [[ "$DELETE_FOLDERS" =~ ^[Yy]$ ]]; then
                for folder in "${EXISTING_FOLDERS[@]}"; do
                    echo "Deleting folder: $folder"
                    rm -rf "$folder"
                done
                echo "All existing multi-gpu folders deleted."
            else
                echo "Keeping existing folders."
            fi
        fi
        
        # Use current date/time for new generation
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATE}"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATE}"
        fi
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
        echo "Starting fresh with clean environment."
    elif [[ "$DATE_OPTION" == "6" ]]; then
        # Custom folder name
        echo ""
        echo "Enter custom folder name (can be any string)"
        echo "Examples:"
        echo "  - 20250729_055123_cu16 (date with suffix)"
        echo "  - hello_world (simple name)"
        echo "  - experiment_v1_beta (descriptive name)"
        echo "  - llama3_8b_test_run (model-specific name)"
        read -p "Custom folder name: " CUSTOM_FOLDER_NAME
        
        # Validate folder name (no spaces, no special characters that could cause issues)
        if [[ "$CUSTOM_FOLDER_NAME" =~ [\ /\\] ]]; then
            echo "Error: Folder name cannot contain spaces, slashes, or backslashes."
            echo "Please use underscores, hyphens, or alphanumeric characters only."
            exit 1
        fi
        
        CUSTOM_DATE="$CUSTOM_FOLDER_NAME"
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_FOLDER_NAME}"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_FOLDER_NAME}"
        fi
        echo "Using custom folder name: $CUSTOM_FOLDER_NAME"
        echo "Folder name: $OUTPUT_DIR"
    else
        # Default: use current date/time
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATE}"
        else
            OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATE}"
        fi
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
    fi
fi

# Check if any existing generated scripts exist in current directory
EXISTING_SCRIPTS=$(ls ${FILE_PREFIX}__gpu*_*.sh 2>/dev/null | wc -l)
if [ $EXISTING_SCRIPTS -gt 0 ] && [ "$OVERWRITE_MODE" != true ]; then
    echo "Found $EXISTING_SCRIPTS existing generated scripts in current directory."
    echo ""
    echo "File handling options:"
    echo "1. Keep old files (default) - new scripts will be generated alongside existing ones"
    echo "2. Delete old files - remove existing scripts and generate new ones"
    echo "3. Overwrite old files - replace existing scripts with new ones"
    read -p "Choose option (1/2/3, default: 1): " FILE_HANDLING_OPTION
    
    if [[ "$FILE_HANDLING_OPTION" == "2" ]]; then
        echo "Deleting existing generated scripts..."
        rm -f ${FILE_PREFIX}__gpu*_*.sh
        rm -f run_all_scripts.sh launch.sh script_summary.txt
        echo "Existing scripts deleted."
    elif [[ "$FILE_HANDLING_OPTION" == "3" ]]; then
        echo "Overwriting existing scripts..."
        # We'll handle the actual overwriting later in the script generation
        OVERWRITE_EXISTING=true
    else
        echo "Keeping existing scripts. New scripts will be generated alongside them."
        OVERWRITE_EXISTING=false
    fi
elif [ $EXISTING_SCRIPTS -gt 0 ] && [ "$OVERWRITE_MODE" == true ]; then
    echo "Overwrite mode: Found $EXISTING_SCRIPTS existing generated scripts in current directory."
    echo "These will be overwritten with new scripts using the same date/time."
    OVERWRITE_EXISTING=true
else
    OVERWRITE_EXISTING=false
fi

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    if [ "$OVERWRITE_MODE" == true ]; then
        echo "Overwrite mode: Output directory $OUTPUT_DIR already exists."
        echo "This directory will be used for regenerating scripts with the same date/time."
    else
        echo "Output directory $OUTPUT_DIR already exists."
        read -p "Do you want to delete it and regenerate? (y/n, default: y): " DELETE_DIR
        if [[ "$DELETE_DIR" =~ ^[Yy]$ ]] || [ -z "$DELETE_DIR" ]; then
            echo "Deleting existing directory: $OUTPUT_DIR"
            rm -rf "$OUTPUT_DIR"
        else
            echo "Keeping existing directory. Will create a new one with different timestamp."
            # Generate a new timestamp to avoid conflict
            if [ -n "$CUSTOM_DATE" ]; then
                # Add a suffix to the custom date to avoid conflict
                if [ "$USE_MODEL_IN_NAMES" = true ]; then
                    OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_${CUSTOM_DATE}_$(date +%H%M%S)"
                else
                    OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_${CUSTOM_DATE}_$(date +%H%M%S)"
                fi
            else
                if [ "$USE_MODEL_IN_NAMES" = true ]; then
                    OUTPUT_DIR="${FILE_PREFIX}__${MODEL_SHORT_NAME}_multi_gpu_$(date +%H%M%S)"
                else
                    OUTPUT_DIR="${FILE_PREFIX}__multi_gpu_$(date +%H%M%S)"
                fi
            fi
            echo "Using new directory name: $OUTPUT_DIR"
        fi
    fi
fi

mkdir -p "$OUTPUT_DIR"
echo ""
echo "Generating scripts in directory: $OUTPUT_DIR"

# Clean up existing scripts in output directory if in overwrite mode
if [ "$OVERWRITE_MODE" == true ] || [ "$OVERWRITE_EXISTING" == true ]; then
    if [ "$OVERWRITE_MODE" == true ]; then
        echo "Overwrite mode: Cleaning up existing scripts in $OUTPUT_DIR..."
    else
        echo "Cleaning up existing scripts in $OUTPUT_DIR for overwrite..."
    fi
    rm -f "$OUTPUT_DIR"/${FILE_PREFIX}__gpu*_*.sh
    rm -f "$OUTPUT_DIR"/run_all_scripts.sh
    rm -f "$OUTPUT_DIR"/launch.sh
    rm -f "$OUTPUT_DIR"/script_summary.txt
    echo "Existing scripts cleaned up."
fi

# Generate scripts
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    desc=$(get_combination_desc $combo)
    
    # Determine GPU assignment
    # Create model short name for filenames
    MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')
    
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        # Custom assignment
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        # For script naming, use all GPUs or a placeholder for -1
        if [[ "$gpu_assignment" == "-1" ]]; then
            gpu_id_display="cpu"
        else
            # Replace commas with underscores for filename compatibility
            gpu_id_display=$(echo "$gpu_assignment" | tr ',' '_')
        fi
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            script_name="${FILE_PREFIX}__${MODEL_SHORT_NAME}_gpu${gpu_id_display}_${desc}_${combo}.sh"
        else
            script_name="${FILE_PREFIX}__gpu${gpu_id_display}_${desc}_${combo}.sh"
        fi
        echo "Generating script $((i+1))/${#COMBINATIONS[@]}: $script_name (GPUs: [$gpu_assignment], $desc, $MODEL_NAME)"
    else
        # Round-robin assignment
        gpu_id=$((i % NUM_GPUS))
        gpu_assignment="$gpu_id"
        if [ "$USE_MODEL_IN_NAMES" = true ]; then
            script_name="${FILE_PREFIX}__${MODEL_SHORT_NAME}_gpu${gpu_id}_${desc}_${combo}.sh"
        else
            script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}_${combo}.sh"
        fi
        echo "Generating script $((i+1))/${#COMBINATIONS[@]}: $script_name (GPU $gpu_id, $desc, $MODEL_NAME)"
    fi
    
    # Read the base script
    cp "${BASE_SCRIPT}.sh" "$OUTPUT_DIR/$script_name"
    
    # Get parameters for this combination
    params=$(get_combination_params $combo)
    read -r RUN_STANDARD_VERSION RUN_OFL_VERSION RUN_STANDARD RUN_CONTEXT_ONLY RUN_MAX_ENTROPY RUN_NO_MAX_ENTROPY <<< "$params"
    
    # Replace the configuration section
    sed -i "s/RUN_STANDARD_VERSION=.*/RUN_STANDARD_VERSION=$RUN_STANDARD_VERSION   # Run standard policy version/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_OFL_VERSION=.*/RUN_OFL_VERSION=$RUN_OFL_VERSION       # Run OFL policy version with enhanced features/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_STANDARD=.*/RUN_STANDARD=$RUN_STANDARD          # Run without --use-context-only-state/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_CONTEXT_ONLY=.*/RUN_CONTEXT_ONLY=$RUN_CONTEXT_ONLY      # Run with --use-context-only-state/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_MAX_ENTROPY=.*/RUN_MAX_ENTROPY=$RUN_MAX_ENTROPY       # Run with max-entropy PPO/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_NO_MAX_ENTROPY=.*/RUN_NO_MAX_ENTROPY=$RUN_NO_MAX_ENTROPY    # Run without max-entropy (standard PPO)/" "$OUTPUT_DIR/$script_name"
    
    # Replace model paths based on selection
    sed -i "s|MODEL_PATH=\".*\"|MODEL_PATH=\"$MODEL_PATH\"|" "$OUTPUT_DIR/$script_name"
    sed -i "s|BASE_MODEL_PATH=\".*\"|BASE_MODEL_PATH=\"$BASE_MODEL_PATH\"|" "$OUTPUT_DIR/$script_name"
    
    # Replace generation script modules for different model types
    sed -i "s/gen_ea_answer_llama3chat_rl/$GEN_SCRIPT/g" "$OUTPUT_DIR/$script_name"
    # Handle the non-RL version replacement carefully
    if [[ "$GEN_SCRIPT" == "gen_ea_answer_vicuna_rl" ]]; then
        sed -i "s/gen_ea_answer_llama3chat/gen_ea_answer_vicuna/g" "$OUTPUT_DIR/$script_name"
    elif [[ "$GEN_SCRIPT" == "gen_ea_answer_llama3chat_rl" ]]; then
        # Keep as is for LLaMA models
        :
    fi
    sed -i "s/gen_baseline_answer_llama3chat/$BASELINE_SCRIPT/g" "$OUTPUT_DIR/$script_name"
    
    # Replace network architecture arguments based on policy version
    if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 0 ]; then
        # Only standard version - replace all --ppo-net-arch with standard architecture
        sed -i "s/--ppo-net-arch \"[^\"]*\"/--ppo-net-arch \"$STANDARD_NET_ARCH\"/g" "$OUTPUT_DIR/$script_name"
    elif [ "$RUN_STANDARD_VERSION" -eq 0 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
        # Only OFL version - replace all --ppo-net-arch with OFL architecture
        sed -i "s/--ppo-net-arch \"[^\"]*\"/--ppo-net-arch \"$OFL_NET_ARCH\"/g" "$OUTPUT_DIR/$script_name"
    elif [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
        # Both versions - need to be more careful about replacement
        # First, replace standard version instances (before OFL version lines)
        sed -i "/--optimized-policy-version standard/,/--ppo-net-arch/s/--ppo-net-arch \"[^\"]*\"/--ppo-net-arch \"$STANDARD_NET_ARCH\"/g" "$OUTPUT_DIR/$script_name"
        # Then, replace OFL version instances (after OFL version lines)
        sed -i "/--optimized-policy-version ofl/,/--ppo-net-arch/s/--ppo-net-arch \"[^\"]*\"/--ppo-net-arch \"$OFL_NET_ARCH\"/g" "$OUTPUT_DIR/$script_name"
    fi
    
    # Replace CUDA_VISIBLE_DEVICES based on GPU assignment
    if [[ "$gpu_assignment" == "-1" ]]; then
        # No GPU - remove CUDA_VISIBLE_DEVICES entirely
        sed -i "s/CUDA_VISIBLE_DEVICES=[0-9,]* //g" "$OUTPUT_DIR/$script_name"
        sed -i "s/python -m/CUDA_VISIBLE_DEVICES= python -m/g" "$OUTPUT_DIR/$script_name"
    else
        # Set CUDA_VISIBLE_DEVICES to the assigned GPUs
        sed -i "s/CUDA_VISIBLE_DEVICES=[0-9,]* //g" "$OUTPUT_DIR/$script_name"
        sed -i "s/python -m/CUDA_VISIBLE_DEVICES=$gpu_assignment python -m/g" "$OUTPUT_DIR/$script_name"
    fi
    
    # Update DATE variable to use the selected date/time
    if [ -n "$CUSTOM_DATE" ]; then
        # Use custom date/time
        sed -i "/^DATE=.*_optimized_ppo/c\DATE=\"${CUSTOM_DATE}_optimized_ppo\"" "$OUTPUT_DIR/$script_name"
    else
        # Use current date/time for simple folder option
        TODAY_DATE=$(date '+%Y%m%d_%H%M%S')
        sed -i "/^DATE=.*_optimized_ppo/c\DATE=\"${TODAY_DATE}_optimized_ppo\"" "$OUTPUT_DIR/$script_name"
    fi
    
    # Store header information to add at the end later
    HEADER_COMMENT=""
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        if [[ "$gpu_assignment" == "-1" ]]; then
            HEADER_COMMENT="# Generated script for CPU only, Combination $combo ($desc)"
        else
            HEADER_COMMENT="# Generated script for GPUs [$gpu_assignment], Combination $combo ($desc)"
        fi
    else
        HEADER_COMMENT="# Generated script for GPU $gpu_id, Combination $combo ($desc)"
    fi
    HEADER_COMMENT="$HEADER_COMMENT\n# Original script: ${BASE_SCRIPT}.sh"
    HEADER_COMMENT="$HEADER_COMMENT\n# Model: $MODEL_NAME ($MODEL_PATH)"
    if [ -n "$CUSTOM_DATE" ]; then
        HEADER_COMMENT="$HEADER_COMMENT\n# Generated on: $CUSTOM_DATE"
    else
        HEADER_COMMENT="$HEADER_COMMENT\n# Generated on: $(date)"
    fi
    # Add network architecture info to header
    if [ "$NEED_STANDARD_VERSION" = true ] && [ "$NEED_OFL_VERSION" = true ]; then
        HEADER_COMMENT="$HEADER_COMMENT\n# Network Architecture - Standard: $STANDARD_NET_ARCH, OFL: $OFL_NET_ARCH"
    elif [ "$NEED_STANDARD_VERSION" = true ]; then
        HEADER_COMMENT="$HEADER_COMMENT\n# Network Architecture - Standard: $STANDARD_NET_ARCH"
    elif [ "$NEED_OFL_VERSION" = true ]; then
        HEADER_COMMENT="$HEADER_COMMENT\n# Network Architecture - OFL: $OFL_NET_ARCH"
    fi
    
    # Substitute model variables in the generated script
    sed -i "s|MODEL_NAME=\".*\"|MODEL_NAME=\"$MODEL_NAME\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|MODEL_PATH=\".*\"|MODEL_PATH=\"$MODEL_PATH\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|BASE_MODEL_PATH=\".*\"|BASE_MODEL_PATH=\"$BASE_MODEL_PATH\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|GEN_SCRIPT=\".*\"|GEN_SCRIPT=\"$GEN_SCRIPT\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|BASELINE_SCRIPT=\".*\"|BASELINE_SCRIPT=\"$BASELINE_SCRIPT\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|EAGLE3_SCRIPT=\".*\"|EAGLE3_SCRIPT=\"$EAGLE3_SCRIPT\"|g" "$OUTPUT_DIR/$script_name"
    
    # Substitute network architecture variables
    sed -i "s|STANDARD_NET_ARCH=\".*\"|STANDARD_NET_ARCH=\"$STANDARD_NET_ARCH\"|g" "$OUTPUT_DIR/$script_name"
    sed -i "s|OFL_NET_ARCH=\".*\"|OFL_NET_ARCH=\"$OFL_NET_ARCH\"|g" "$OUTPUT_DIR/$script_name"
    
    # Update DATE to include model name for better organization (if enabled)
    if [ "$USE_MODEL_IN_NAMES" = true ]; then
        if [ -n "$CUSTOM_DATE" ]; then
            sed -i "s|DATE=\".*\"|DATE=\"${CUSTOM_DATE}_${MODEL_SHORT_NAME}\"|g" "$OUTPUT_DIR/$script_name"
        else
            current_date=$(date '+%Y%m%d_%H%M%S')
            sed -i "s|DATE=\$(date.*)|DATE=\"${current_date}_${MODEL_SHORT_NAME}\"|g" "$OUTPUT_DIR/$script_name"
        fi
    else
        # Use legacy DATE format with "_optimized_ppo" suffix
        if [ -n "$CUSTOM_DATE" ]; then
            sed -i "s|DATE=\".*\"|DATE=\"${CUSTOM_DATE}_optimized_ppo\"|g" "$OUTPUT_DIR/$script_name"
        else
            current_date=$(date '+%Y%m%d_%H%M%S')
            sed -i "s|DATE=\$(date.*)|DATE=\"${current_date}_optimized_ppo\"|g" "$OUTPUT_DIR/$script_name"
        fi
    fi
    
    # Add header comments at the end of the script
    echo "" >> "$OUTPUT_DIR/$script_name"
    echo -e "$HEADER_COMMENT" >> "$OUTPUT_DIR/$script_name"
done

# Create a master script to run all generated scripts
MASTER_SCRIPT="$OUTPUT_DIR/run_all_scripts.sh"
echo "#!/bin/bash" > "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"
echo "# Master script to run all generated GPU scripts" >> "$MASTER_SCRIPT"
echo "# Model: $MODEL_NAME ($MODEL_PATH)" >> "$MASTER_SCRIPT"
if [ -n "$CUSTOM_DATE" ]; then
    echo "# Generated on: $CUSTOM_DATE" >> "$MASTER_SCRIPT"
else
    echo "# Generated on: $(date)" >> "$MASTER_SCRIPT"
fi
echo "# Number of scripts: ${#COMBINATIONS[@]}" >> "$MASTER_SCRIPT"
echo "# Available GPUs: $NUM_GPUS" >> "$MASTER_SCRIPT"
if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    echo "# GPU Assignment: Custom" >> "$MASTER_SCRIPT"
    for i in "${!COMBINATIONS[@]}"; do
        combo=${COMBINATIONS[$i]}
        desc=$(get_combination_desc $combo)
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            echo "#   Script $((i+1)) (Combination $combo): CPU only" >> "$MASTER_SCRIPT"
        else
            echo "#   Script $((i+1)) (Combination $combo): GPUs [$gpu_assignment]" >> "$MASTER_SCRIPT"
        fi
    done
else
    echo "# GPU Assignment: Round-robin across $NUM_GPUS GPUs" >> "$MASTER_SCRIPT"
fi
# Add network architecture info to master script header
if [ "$NEED_STANDARD_VERSION" = true ] && [ "$NEED_OFL_VERSION" = true ]; then
    echo "# Network Architecture - Standard: $STANDARD_NET_ARCH, OFL: $OFL_NET_ARCH" >> "$MASTER_SCRIPT"
elif [ "$NEED_STANDARD_VERSION" = true ]; then
    echo "# Network Architecture - Standard: $STANDARD_NET_ARCH" >> "$MASTER_SCRIPT"
elif [ "$NEED_OFL_VERSION" = true ]; then
    echo "# Network Architecture - OFL: $OFL_NET_ARCH" >> "$MASTER_SCRIPT"
fi
echo "" >> "$MASTER_SCRIPT"

# Group scripts by GPU or create individual execution
declare -A gpu_scripts
declare -a individual_scripts

for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    desc=$(get_combination_desc $combo)
    
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        # Custom assignment - each script runs independently
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            gpu_id_display="cpu"
        else
            IFS=',' read -ra GPUS <<< "$gpu_assignment"
            gpu_id_display="${GPUS[0]}"
        fi
        script_name="${FILE_PREFIX}__gpu${gpu_id_display}_${desc}_${combo}.sh"
        individual_scripts+=("$script_name")
    else
        # Round-robin assignment - group by GPU
        gpu_id=$((i % NUM_GPUS))
        script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}_${combo}.sh"
        
        if [ -z "${gpu_scripts[$gpu_id]}" ]; then
            gpu_scripts[$gpu_id]=""
        fi
        gpu_scripts[$gpu_id]="${gpu_scripts[$gpu_id]} $script_name"
    fi
done

# Add GPU-specific sections to master script
echo "echo \"=== EAGLE RL Multi-GPU Training Started ===\"" >> "$MASTER_SCRIPT"
echo "echo \"Start time: \$(date)\"" >> "$MASTER_SCRIPT"
echo "echo \"Number of scripts: ${#COMBINATIONS[@]}\"" >> "$MASTER_SCRIPT"
echo "echo \"Available GPUs: $NUM_GPUS\"" >> "$MASTER_SCRIPT"
echo "echo \"\"" >> "$MASTER_SCRIPT"

# Create a function to run scripts on each GPU
echo "run_gpu_scripts() {" >> "$MASTER_SCRIPT"
echo "    local gpu_id=\$1" >> "$MASTER_SCRIPT"
echo "    shift" >> "$MASTER_SCRIPT"
echo "    local scripts=(\"\$@\")" >> "$MASTER_SCRIPT"
echo "    echo \"=== Starting scripts for GPU \$gpu_id ===\"" >> "$MASTER_SCRIPT"
echo "    for script in \"\${scripts[@]}\"; do" >> "$MASTER_SCRIPT"
echo "        echo \"Running \$script on GPU \$gpu_id...\"" >> "$MASTER_SCRIPT"
echo "        bash \"\$script\" &" >> "$MASTER_SCRIPT"
echo "    done" >> "$MASTER_SCRIPT"
echo "    wait" >> "$MASTER_SCRIPT"
echo "    echo \"=== Completed all scripts for GPU \$gpu_id ===\"" >> "$MASTER_SCRIPT"
echo "}" >> "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"

# Run scripts based on assignment method
if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    # Custom assignment - run each script individually
    echo "echo \"Running ${#individual_scripts[@]} scripts with custom GPU assignments...\"" >> "$MASTER_SCRIPT"
    for script in "${individual_scripts[@]}"; do
        echo "echo \"Starting \$script...\"" >> "$MASTER_SCRIPT"
        echo "bash \"\$script\" &" >> "$MASTER_SCRIPT"
    done
else
    # Round-robin assignment - run scripts grouped by GPU
    for gpu_id in "${!gpu_scripts[@]}"; do
        echo "run_gpu_scripts $gpu_id ${gpu_scripts[$gpu_id]} &" >> "$MASTER_SCRIPT"
    done
fi

echo "" >> "$MASTER_SCRIPT"
echo "# Wait for all GPUs to complete" >> "$MASTER_SCRIPT"
echo "wait" >> "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"
echo "echo \"\"" >> "$MASTER_SCRIPT"
echo "echo \"=== All GPU scripts completed ===\"" >> "$MASTER_SCRIPT"
echo "echo \"End time: \$(date)\"" >> "$MASTER_SCRIPT"

chmod +x "$MASTER_SCRIPT"

# Create a simple launcher script
LAUNCHER_SCRIPT="$OUTPUT_DIR/launch.sh"
echo "#!/bin/bash" > "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "# Simple launcher for EAGLE RL Multi-GPU Training" >> "$LAUNCHER_SCRIPT"
echo "# Model: $MODEL_NAME ($MODEL_PATH)" >> "$LAUNCHER_SCRIPT"
echo "# Usage: bash launch.sh" >> "$LAUNCHER_SCRIPT"
if [ -n "$CUSTOM_DATE" ]; then
    echo "# Generated on: $CUSTOM_DATE" >> "$LAUNCHER_SCRIPT"
else
    echo "# Generated on: $(date)" >> "$LAUNCHER_SCRIPT"
fi
echo "" >> "$LAUNCHER_SCRIPT"
echo "echo \"=== EAGLE RL Multi-GPU Training Launcher ===\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"Start time: \$(date)\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"\"" >> "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "# Run the master script" >> "$LAUNCHER_SCRIPT"
echo "bash run_all_scripts.sh" >> "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "echo \"\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"=== Training completed ===\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"End time: \$(date)\"" >> "$LAUNCHER_SCRIPT"
chmod +x "$LAUNCHER_SCRIPT"

# Create a summary file
SUMMARY_FILE="$OUTPUT_DIR/script_summary.txt"
echo "EAGLE RL Multi-GPU Script Generation Summary" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
if [ -n "$CUSTOM_DATE" ]; then
    echo "Generated on: $CUSTOM_DATE" >> "$SUMMARY_FILE"
else
    echo "Generated on: $(date)" >> "$SUMMARY_FILE"
fi
echo "Base script: ${BASE_SCRIPT}.sh" >> "$SUMMARY_FILE"
echo "Model: $MODEL_NAME ($MODEL_PATH)" >> "$SUMMARY_FILE"
echo "Base model: $BASE_MODEL_PATH" >> "$SUMMARY_FILE"
echo "Generation script: $GEN_SCRIPT" >> "$SUMMARY_FILE"
echo "Baseline script: $BASELINE_SCRIPT" >> "$SUMMARY_FILE"
echo "Number of GPUs: $NUM_GPUS" >> "$SUMMARY_FILE"
echo "Number of scripts: ${#COMBINATIONS[@]}" >> "$SUMMARY_FILE"
if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    echo "GPU Assignment Method: Custom" >> "$SUMMARY_FILE"
else
    echo "GPU Assignment Method: Round-robin" >> "$SUMMARY_FILE"
fi
echo "" >> "$SUMMARY_FILE"
echo "Network Architecture Configuration:" >> "$SUMMARY_FILE"
if [ "$NEED_STANDARD_VERSION" = true ]; then
    echo "  Standard Version: $STANDARD_NET_ARCH" >> "$SUMMARY_FILE"
fi
if [ "$NEED_OFL_VERSION" = true ]; then
    echo "  OFL Version: $OFL_NET_ARCH" >> "$SUMMARY_FILE"
fi
echo "" >> "$SUMMARY_FILE"
if [ "$COMBINATION_MODE" -eq 1 ]; then
    echo "Mode: Pre-built combinations" >> "$SUMMARY_FILE"
    echo "Selected combinations:" >> "$SUMMARY_FILE"
else
    echo "Mode: Custom parameter combinations" >> "$SUMMARY_FILE"
    echo "Custom parameter combinations:" >> "$SUMMARY_FILE"
fi
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    desc=$(get_combination_desc $combo)
    
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            gpu_id_display="cpu"
        else
            IFS=',' read -ra GPUS <<< "$gpu_assignment"
            gpu_id_display="${GPUS[0]}"
        fi
        script_name="${FILE_PREFIX}__gpu${gpu_id_display}_${desc}_${combo}.sh"
        if [ "$COMBINATION_MODE" -eq 1 ]; then
            echo "  Script $((i+1)): $script_name (GPUs: [$gpu_assignment], $desc)" >> "$SUMMARY_FILE"
        else
            params=$(get_combination_params $combo)
            echo "  Script $((i+1)): $script_name (GPUs: [$gpu_assignment], $desc, params: $params)" >> "$SUMMARY_FILE"
        fi
    else
        gpu_id=$((i % NUM_GPUS))
        script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}_${combo}.sh"
        if [ "$COMBINATION_MODE" -eq 1 ]; then
            echo "  Script $((i+1)): $script_name (GPU $gpu_id, $desc)" >> "$SUMMARY_FILE"
        else
            params=$(get_combination_params $combo)
            echo "  Script $((i+1)): $script_name (GPU $gpu_id, $desc, params: $params)" >> "$SUMMARY_FILE"
        fi
    fi
done
echo "" >> "$SUMMARY_FILE"
if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    echo "Custom GPU assignments:" >> "$SUMMARY_FILE"
    for i in "${!COMBINATIONS[@]}"; do
        combo=${COMBINATIONS[$i]}
        desc=$(get_combination_desc $combo)
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            echo "  Script $((i+1)) (Combination $combo): CPU only" >> "$SUMMARY_FILE"
        else
            echo "  Script $((i+1)) (Combination $combo): GPUs [$gpu_assignment]" >> "$SUMMARY_FILE"
        fi
    done
else
    echo "GPU distribution (round-robin):" >> "$SUMMARY_FILE"
    for gpu_id in "${!gpu_scripts[@]}"; do
        echo "  GPU $gpu_id: ${gpu_scripts[$gpu_id]}" >> "$SUMMARY_FILE"
    done
fi
echo "" >> "$SUMMARY_FILE"
echo "To run all scripts: bash $MASTER_SCRIPT" >> "$SUMMARY_FILE"
echo "To run individual scripts: bash <script_name>" >> "$SUMMARY_FILE"
echo "To run with launcher: bash launch.sh" >> "$SUMMARY_FILE"

echo ""
echo "=== Generation Complete ==="
if [ "$OVERWRITE_MODE" == true ]; then
    echo "Regenerated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (overwrite mode)"
    echo "Date/time preserved for training continuity: $CUSTOM_DATE"
elif [ "$OVERWRITE_EXISTING" == true ]; then
    echo "Regenerated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (overwrite existing)"
    if [ -n "$CUSTOM_DATE" ]; then
        echo "Date/time preserved for training continuity: $CUSTOM_DATE"
    fi
elif [[ "$DATE_OPTION" == "5" ]]; then
    echo "Generated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (fresh start)"
    echo "Old files cleaned up, starting with clean environment."
else
    echo "Generated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR"
fi
echo ""
echo "Scripts created:"
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    desc=$(get_combination_desc $combo)
    
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            gpu_id_display="cpu"
        else
            IFS=',' read -ra GPUS <<< "$gpu_assignment"
            gpu_id_display="${GPUS[0]}"
        fi
        script_name="${FILE_PREFIX}__gpu${gpu_id_display}_${desc}_${combo}.sh"
        if [[ "$gpu_assignment" == "-1" ]]; then
            echo "  $script_name (CPU only, $desc)"
        else
            echo "  $script_name (GPUs: [$gpu_assignment], $desc)"
        fi
    else
        gpu_id=$((i % NUM_GPUS))
        script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}_${combo}.sh"
        echo "  $script_name (GPU $gpu_id, $desc)"
    fi
done
echo ""
echo "Master script: $MASTER_SCRIPT"
echo "Launcher script: $LAUNCHER_SCRIPT"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "Configuration Summary:"
if [ "$COMBINATION_MODE" -eq 1 ]; then
    echo "  Combination Mode: Pre-built combinations"
else
    echo "  Combination Mode: Custom parameter combinations"
fi
echo "  Selected Model: $MODEL_NAME"
echo "  Model Path: $MODEL_PATH"
echo "  Base Model Path: $BASE_MODEL_PATH"
echo "  Generation Script: $GEN_SCRIPT"
echo ""
echo "Network Architecture Configuration:"
if [ "$NEED_STANDARD_VERSION" = true ]; then
    echo "  Standard Version: $STANDARD_NET_ARCH"
fi
if [ "$NEED_OFL_VERSION" = true ]; then
    echo "  OFL Version: $OFL_NET_ARCH"
fi
echo ""
echo "GPU Assignment Configuration:"
if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
    echo "  Method: Custom GPU assignment"
    for i in "${!COMBINATIONS[@]}"; do
        combo=${COMBINATIONS[$i]}
        desc=$(get_combination_desc $combo)
        gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
        if [[ "$gpu_assignment" == "-1" ]]; then
            echo "    Script $((i+1)) (Combination $combo): CPU only"
        else
            echo "    Script $((i+1)) (Combination $combo): GPUs [$gpu_assignment]"
        fi
    done
else
    echo "  Method: Round-robin across $NUM_GPUS GPUs"
fi
echo ""
echo "To run all scripts: bash $MASTER_SCRIPT"
echo "To run with launcher: bash launch.sh"
echo "To run individual scripts: cd $OUTPUT_DIR && bash <script_name>"

# Ask user if they want to run the scripts now
echo ""
read -p "Do you want to run all generated scripts now? (y/n, default: n): " RUN_NOW

if [[ "$RUN_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Starting execution of all scripts ==="
    echo "Changing to output directory: $OUTPUT_DIR"
    cd "$OUTPUT_DIR"
    
    echo ""
    echo "Running master script: $MASTER_SCRIPT"
    if [ "$GPU_ASSIGNMENT_METHOD" -eq 2 ]; then
        echo "This will execute all ${#COMBINATIONS[@]} scripts with custom GPU assignments"
        for i in "${!COMBINATIONS[@]}"; do
            combo=${COMBINATIONS[$i]}
            desc=$(get_combination_desc $combo)
            gpu_assignment="${GPU_ASSIGNMENTS[$i]}"
            if [[ "$gpu_assignment" == "-1" ]]; then
                echo "  Script $((i+1)) (Combination $combo): CPU only"
            else
                echo "  Script $((i+1)) (Combination $combo): GPUs [$gpu_assignment]"
            fi
        done
    else
        echo "This will execute all ${#COMBINATIONS[@]} scripts across $NUM_GPUS GPUs (round-robin)"
    fi
    echo "Press Ctrl+C to stop execution"
    echo ""
    
    # Run the master script
    bash "$MASTER_SCRIPT"
    
    echo ""
    echo "=== Execution completed ==="
    echo "Check the log files in $OUTPUT_DIR for results"
else
    echo ""
    echo "Scripts generated but not executed."
    echo "To run later: cd $OUTPUT_DIR && bash launch.sh"
    echo "Or run individual scripts: cd $OUTPUT_DIR && bash <script_name>"
    
    # Add note about file handling if scripts were kept
    if [ "$OVERWRITE_EXISTING" == false ] && [ $EXISTING_SCRIPTS -gt 0 ]; then
        echo ""
        echo "Note: Existing scripts were kept. New scripts are generated alongside them."
        echo "You may want to organize or clean up the directory manually."
    fi
fi

# Add helpful note about overwrite functionality
if [ "$OVERWRITE_MODE" == true ] || [ "$OVERWRITE_EXISTING" == true ]; then
    echo ""
    echo "=== Overwrite Information ==="
    if [ -n "$CUSTOM_DATE" ]; then
        echo "Scripts were regenerated with the same date/time: $CUSTOM_DATE"
        echo "This allows you to resume training from where you left off."
        echo "The DATE variable in all scripts remains consistent."
        echo "Log directories will use the same timestamp for continuity."
    else
        echo "Scripts were regenerated with overwrite mode."
        echo "Existing scripts have been replaced with new ones."
    fi
fi 