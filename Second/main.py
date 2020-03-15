import os
import numpy as np
import sys

TEST_PATH = r"C:\Users\dorvic\PycharmProjects\psiml-2019\Second\Test"
TEST_NUMBERS = [3,4,5,6,7,8,9,10,11,12]
#TEST_NUMBERS = [4]
ANSWERS_EXT = ".txt"
CORRECT_ANS_ID = 0
WISE_PER_ANS_ID = 1
EPSILON = 0.01
EPSILON_THRESHOLD = 0.00000001
TASK_THRESHOLD = 70

def get_test_name(file_name: str, start_str: str):
    return file_name.lstrip(start_str).rstrip(ANSWERS_EXT)


def count_valid_answers(dic_test_name: dict, test_name: str, start_idx: int):
    if test_name not in dic_test_name:
        init_array = [None, None]
    else:
        init_array = dic_test_name[test_name]
    dic_test_name[test_name] = init_array


def is_answer_positive(file_path: str):
    with open(file_path) as f:
        first_line = f.readline()
        return first_line == "Yes"


def wise_person_certianity(file_path: str):
    with open(file_path) as f:
        first_line = f.readline()
        first_line = first_line.rstrip("%")
        return int(first_line)


def update_correct_answer(dict_test_name: dict, file_path: str, test_name: str):
    correct_answer = is_answer_positive(file_path)
    dict_test_name[test_name][CORRECT_ANS_ID] = correct_answer


def update_wise_person_certanity(dict_test_name: dict, file_path: str, test_name: str):
    certainty = wise_person_certianity(file_path)
    dict_test_name[test_name][WISE_PER_ANS_ID] = certainty


def generate_all_subdirs(dir_name_str: str):
    sub_dirs  = []
    dirs_to_visit_a = [dir_name_str]
    while dirs_to_visit_a.__len__() > 0:
        dir_to_visit = dirs_to_visit_a.pop(0)
        sub_dirs.append(dir_to_visit)
        sub_dirs_to_visit = [f.path for f in os.scandir(dir_to_visit) if f.is_dir()]
        dirs_to_visit_a = dirs_to_visit_a + sub_dirs_to_visit
    return sub_dirs

def parse_folder(dir_name_str: str, start_str_a):
    dic_test_name = {}
    #global_cnt = 0
    sub_dirs = generate_all_subdirs(dir_name_str)
    for sub_dir in sub_dirs:
        directory = os.fsencode(sub_dir)
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if not file_name.endswith(ANSWERS_EXT):
                continue
            #global_cnt += 1
            for start_idx, start_str in enumerate(start_str_a):
                if file_name.startswith(start_str):

                    test_name = get_test_name(file_name, start_str)
                    #Invalid test name (not number)
                    if not test_name.isdigit():
                        continue

                    file_path = os.path.join(sub_dir, file_name)
                    count_valid_answers(dic_test_name, test_name, start_idx)
                    if start_idx == CORRECT_ANS_ID:
                        update_correct_answer(dic_test_name, file_path, test_name)
                    elif start_idx == WISE_PER_ANS_ID:
                        update_wise_person_certanity(dic_test_name, file_path, test_name)
                    else:
                        pass
    #print(global_cnt)
    return dic_test_name


def count_positive_and_negative_answers(dic_test_name: dict):
    correct_answer_postive = 0
    correct_answer_negative = 0
    for it in dic_test_name.values():
        correct_answer = it[CORRECT_ANS_ID]
        if correct_answer is not None:
            correct_answer_postive += correct_answer
            correct_answer_negative += int(not correct_answer)
    return correct_answer_postive, correct_answer_negative


def count_valid_tests(dic_test_name: dict):
    valid_tests = 0
    for it in dic_test_name.values():
        if it[CORRECT_ANS_ID] is not None and it[WISE_PER_ANS_ID] is not None:
            valid_tests += 1
    return valid_tests


def count_true_positive(dic_test_name: dict, threshold: float):
    true_positive = 0.0
    for it in dic_test_name.values():
        wp_certainty =  it[WISE_PER_ANS_ID]
        correct_answer = it[CORRECT_ANS_ID]
        if (correct_answer is not None) and (wp_certainty is not None) and \
            correct_answer and (wp_certainty >= threshold):
            true_positive += 1

    return true_positive


def count_false_positive(dic_test_name: dict, threshold: float):
    false_positive = 0.0
    for it in dic_test_name.values():
        wp_certainty = it[WISE_PER_ANS_ID]
        correct_answer = it[CORRECT_ANS_ID]
        if (correct_answer is not None) and (wp_certainty is not None) and \
                (not correct_answer) and (wp_certainty >= threshold):
            false_positive += 1

    return false_positive


def remove_invalid_tests(dic_test_name: dict):
    dic_test_name_c = dic_test_name
    dic_test_name = {}
    for key, value in dic_test_name_c.items():
        if value[CORRECT_ANS_ID] is None or value[WISE_PER_ANS_ID] is None:
            continue
        dic_test_name[key] = value
    return dic_test_name


def compute_eer(dic_test_name: dict):
    threshold = 0.0
    threshold_begin = 0.0
    threshold_end = 100.0
    correct_answer_positive, correct_answer_negative = \
        count_positive_and_negative_answers(dic_test_name)

    epsilon_threshold = 10
    threshold_nums = 11
    threshold = threshold_begin

    sol_found = False
    threshold_a = np.linspace(threshold_begin, threshold_end, threshold_nums)
    while True:
        diff_a = [sys.float_info.max] * threshold_nums
        fpr_a = [sys.float_info.max] * threshold_nums
        for threshold_idx, threshold in enumerate(threshold_a):
            true_positive = count_true_positive(dic_test_name, threshold)
            false_positive = count_false_positive(dic_test_name, threshold)
            tpr = None
            fpr = None
            # 1 - tpr goes up, since number of yess will drop down for sure
            # It always starts from 0, because all positive answers will be predicted true,
            # and complement of that
            # fpr goes down
            # It always starts from 1, because all negative answers will be predicted true
            if correct_answer_positive != 0:
                tpr = true_positive / correct_answer_positive
            if correct_answer_negative != 0:
                fpr = false_positive / correct_answer_negative

            if (fpr is not None) and (tpr is not None):
                diff_a[threshold_idx] = fpr - (1 - tpr)
                fpr_a[threshold_idx] = fpr
        # Check whether the difference is so small to fit.
        start_diff_idx = -1
        for diff_idx, diff in enumerate(diff_a):
            #print("THREASHOLD: {2:.4f} TPR:{0:.5f} FPR{1:.5f} DIF: {3:.4f}".format(tpr, fpr_a, threshold, diff))
            if abs(diff) < EPSILON:
                #print("EER{}".format(fpr))
                print("{}".format(fpr_a[diff_idx]), end="")
                sol_found = True
                break
            # Breaking point
            if diff < 0.0:
                start_diff_idx = diff_idx - 1
                break

        if sol_found:
            break

        epsilon_threshold = threshold_a[threshold_nums - 1] - threshold_a[0]
        #print("EPS_THR{}".format(epsilon_threshold))
        if epsilon_threshold < EPSILON_THRESHOLD:
            #print("EER{}".format(fpr_a[0]))
            print("{}".format(fpr_a[0]), end="")
            break

        threshold_a = np.linspace(threshold_a[start_diff_idx], threshold_a[start_diff_idx+1], threshold_nums)


def compute_eer_b(dic_test_name: dict):
    threshold = 0.0
    threshold_begin = 0.0
    threshold_end = 100.0
    correct_answer_positive, correct_answer_negative = \
        count_positive_and_negative_answers(dic_test_name)

    epsilon_threshold = 10
    threshold_nums = 201
    threshold = threshold_begin

    sol_found = False
    threshold_a = np.linspace(threshold_begin, threshold_end, threshold_nums)
    while True:
        diff_a = [sys.float_info.max] * threshold_nums
        fpr_a = [sys.float_info.max] * threshold_nums
        for threshold_idx, threshold in enumerate(threshold_a):
            true_positive = count_true_positive(dic_test_name, threshold)
            false_positive = count_false_positive(dic_test_name, threshold)
            tpr = None
            fpr = None
            # 1 - tpr goes up, since number of yess will drop down for sure
            # It always starts from 0, because all positive answers will be predicted true,
            # and complement of that
            # fpr goes down
            # It always starts from 1, because all negative answers will be predicted true
            if correct_answer_positive != 0:
                tpr = true_positive / correct_answer_positive
            if correct_answer_negative != 0:
                fpr = false_positive / correct_answer_negative

            if (fpr is not None) and (tpr is not None):
                diff_a[threshold_idx] = fpr - (1 - tpr)
                fpr_a[threshold_idx] = fpr
        # Check whether the difference is so small to fit.
        start_diff_idx = -1
        min_sol = sys.float_info.max
        min_fpr = -1
        for diff_idx, diff in enumerate(diff_a):
            #print("THREASHOLD: {2:.4f} TPR:{0:.5f} FPR{1:.5f} DIF: {3:.4f}".format(tpr, fpr_a, threshold, diff))
            if abs(diff) < min_sol:
                min_fpr = fpr_a[diff_idx]
                min_sol = abs(diff)
                print("THRE{} OTHER{}".format( threshold_a[diff_idx], diff))

                #print("EER{}".format(fpr))
                #print("{}".format(fpr), end="")
                #sol_found = True
                #break
            # Breaking point
            #if diff < 0.0:
                #start_diff_idx = diff_idx - 1
                #break

        print(min_fpr, end="")
        print("min_sol{}".format(min_sol))
        break
        #if sol_found:
        #    break

        #epsilon_threshold = threshold_a[threshold_nums - 1] - threshold_a[0]
        #print("EPS_THR{}".format(epsilon_threshold))
        #if epsilon_threshold < EPSILON_THRESHOLD:
            #print("EER{}".format(fpr_a[0]))
        #    print("{}".format(fpr_a[0]), end="")
        #    break

        #threshold_a = np.linspace(threshold_a[start_diff_idx], threshold_a[start_diff_idx+1], threshold_nums)


# Returns a dictionary where the key is the test id and a value is an array of ca/wa
def list_all_files_start_with_in_dir(dir_name_str: str, start_str_a):
    dic_test_name = parse_folder(dir_name_str, start_str_a)
    #print(dic_test_name)
    correct_answer_positive, correct_answer_negative = \
        count_positive_and_negative_answers(dic_test_name)
    valid_tests = count_valid_tests(dic_test_name)

    #print("Positive{}".format(correct_answer_positive))
    #print("Negative{}".format(correct_answer_negative))
    #print("Valid tests{}".format(valid_tests))
    print("{},{},{},".format(correct_answer_positive, correct_answer_negative, valid_tests), end="")
    dic_test_name = remove_invalid_tests(dic_test_name)
    correct_answer_positive, correct_answer_negative = \
        count_positive_and_negative_answers(dic_test_name)
    valid_tests = count_valid_tests(dic_test_name)
    #print(dic_test_name)
    #print("Positive{}".format(correct_answer_positive))
    #print("Negative{}".format(correct_answer_negative))
    #print("Valid tests{}".format(valid_tests))

    true_positive = count_true_positive(dic_test_name, TASK_THRESHOLD)
    false_positive = count_false_positive(dic_test_name, TASK_THRESHOLD)
    if (correct_answer_positive != 0):
        #print("True positive rate{}".format(true_positive / correct_answer_positive))
        print("{},".format(true_positive / correct_answer_positive), end="")
    if (correct_answer_negative != 0):
        #print("False positive rate{}".format(false_positive / correct_answer_negative))
        print("{},".format(false_positive / correct_answer_negative), end="")
    compute_eer(dic_test_name)


def run_tests():
    answers_folder_a = []
    for test_number in TEST_NUMBERS:
        answers_folder_a.append(TEST_PATH + str(test_number))

    # Run test cases.
    for answers_folder in answers_folder_a:
        print("Test path:" + answers_folder)
        list_all_files_start_with_in_dir(answers_folder, ANS_STR_A)
        print("")


if __name__ == "__main__":
    ANS_STR_A = ["", ""]
    ANS_STR_A[WISE_PER_ANS_ID] = "wpa"
    ANS_STR_A[CORRECT_ANS_ID] = "ca"

    #run_tests()
    answers_folder = input()
    list_all_files_start_with_in_dir(answers_folder, ANS_STR_A)


