import os
import argparse

DATA_DRIVE = '/iblsn/data/Arjun/Plants'
DATASETS_DIR = '%s/datasets' % DATA_DRIVE
CATEGORIES_DIR = '%s/plant_categories' % DATA_DRIVE

def plant_info(plant_name):
    plant_info = plant_name.split('_')
    species = plant_info[0]
    condition = plant_info[1]
    replicate = plant_info[2]
    day = plant_info[3]
    day = day[1:]
    day = int(day)

    return species, condition, replicate, day

def write_categories():
    with open('%s/plant_categories.csv' % CATEGORIES_DIR, 'w') as f:
        f.write('plant, species, condition, replicate, day\n')
        for plant_file in os.listdir(DATASETS_DIR):
            plant_name = plant_file[:-7]
            species, condition, replicate, day = plant_info(plant_name)
            f.write('%s, %s, %s, %s, %d\n' % (plant_name, species, condition, replicate, day))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--write', action='store_true')

    args = parser.parse_args()
    write = args.write

    if write:
        write_categories()

if __name__ == '__main__':
    main()
