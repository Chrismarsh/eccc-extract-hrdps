from convert import extract_day
import pickle
import sys

def main(file):
    with open(file, 'rb') as f:
        loaded_array = pickle.load(f)

    for itr in loaded_array:
        year, month, day = itr
        extract_day(year, month, day)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
        main(file)
    else:
        print("No argument provided")