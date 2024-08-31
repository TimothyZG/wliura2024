#!/bin/bash
#SBATCH --mem=8000M                    # Memory required for the job (adjusted down since downloading doesn't require much)
#SBATCH --nodes=1                      # Only one node is required
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=1              # Only one CPU needed for downloading
#SBATCH --time=02:00:00                # Time limit (set to 1 hour, adjust as needed)
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=recipe/rxrx1/output/download-rxrx1-%j.out  # Output log file

# Load Python environment
module purge
module load python/3.10                # Load Python module
source ~/py310/bin/activate            # Activate virtual environment

# Print current working directory and list contents
echo "Current working directory:"
pwd
echo "Listing available directories and files:"
ls

# Run the Python download script
python recipe/rxrx1/download-rxrx1.py