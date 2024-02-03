"""
Used to check large csv file headers without opening the full document
sample:
python3 HACKING/csvHeaderPeek.py datasets/PatEx_Full\?/
OR:
python3 HACKING/csvHeaderPeek.py datasets/PatEx_Full\?/transactions.csv
"""

import os
import argparse
import pandas as pd

def read_first_lines(input_path, num_lines):
    try:
        if os.path.isdir(input_path):
            for file_name in os.listdir(input_path):
                file_path = os.path.join(input_path, file_name)
                if os.path.isfile(file_path):
                    print(f"{file_name}:")
                    try:
                        df = pd.read_csv(file_path, header=None, nrows=6)
                        print(df.head(6).to_string(index=False, header=False))
                    except:
                        with open(file_path, 'r') as file:
                            for _ in range(num_lines):
                                line = file.readline()
                                if not line:
                                    break
                                print(line.strip())
                    print("\n")
        elif os.path.isfile(input_path):
            print(f"\n{os.path.basename(input_path)}:")
            with open(input_path, 'r') as file:
                for _ in range(num_lines):
                    line = file.readline()
                    if not line:
                        break
                    print(line.strip())
        else:
            print("Invalid path.")
    except FileNotFoundError:
        print("File or directory not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the first lines of a file or every file in a directory.")
    parser.add_argument("input_path", type=str, help="Path to the file or directory.")
    parser.add_argument("--num_lines", type=int, default=6, help="Number of lines to read from each file (default: 6).")
    args = parser.parse_args()

    read_first_lines(args.input_path, args.num_lines)
