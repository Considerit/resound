import os
from utilities import conf
import glob


current_locks = {}


def is_locked(lock_str):
    full_output_dir = conf.get("temp_directory")
    lock_file = os.path.join(full_output_dir, f"locked-{lock_str}")
    if os.path.exists(lock_file):
        return True
    return False


def request_lock(lock_str):
    full_output_dir = conf.get("temp_directory")
    lock_file = os.path.join(full_output_dir, f"locked-{lock_str}")
    if os.path.exists(lock_file):
        return False

    if lock_str == "compilation":
        if len(other_locks()) > 0:
            return False

    global current_locks
    lock = open(lock_file, "w")
    lock.write(f"yo")
    lock.close()
    current_locks[lock_str] = True

    return True


def free_lock(lock_str):
    global current_locks
    full_output_dir = conf.get("temp_directory")
    lock_file = os.path.join(full_output_dir, f"locked-{lock_str}")
    if os.path.exists(lock_file):
        os.remove(lock_file)
    del current_locks[lock_str]


def free_all_locks():
    global current_locks
    locks = list(current_locks.keys())
    for lock in locks:
        free_lock(lock)


def other_locks():
    global current_locks
    full_output_dir = conf.get("temp_directory")
    # Use glob to get all files that match the pattern "locked-*"
    lock_files = glob.glob(os.path.join(full_output_dir, "locked-*"))

    # Strip off the "locked-" prefix and directory structure for each file
    stripped_files = [os.path.basename(f)[7:] for f in lock_files]

    return [lock for lock in stripped_files if lock not in current_locks]
