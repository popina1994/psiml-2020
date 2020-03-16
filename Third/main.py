import json
import os
import numpy as np
import sys

TEST_PATH = r"C:\Users\dorvic\PycharmProjects\psiml-2019\Third" + "\\"
#TEST_NUMBERS = range(0,8) + range(10,16)
TEST_NUMBERS = [8]
#TEST_NUMBERS = range(16)
X_ID = 0
Y_ID = 1
HEIGHT_ID = 2
WEIGHT_ID = 3


def read_json_data(input_path: str):
    with open(input_path) as f:
        data = json.load(f)
    return data


def parse_json_frames_bb(frames_json):
    frames_dict = {}
    for frame_comp in frames_json:
        frame_index = frame_comp["frame_index"]
        identity_bounding_boxes = frame_comp["bounding_boxes"]
        dict_identities = {}
        for identity in identity_bounding_boxes:
            array_coordinates = [0] * 4
            array_coordinates[HEIGHT_ID] = identity['bounding_box']['h']
            array_coordinates[WEIGHT_ID] = identity['bounding_box']['w']
            array_coordinates[X_ID] = identity['bounding_box']['x']
            array_coordinates[Y_ID] = identity['bounding_box']['y']
            if identity["identity"] in dict_identities:
                print("ERRROR")
            dict_identities[identity["identity"]] = array_coordinates
        frames_dict[frame_index] = dict_identities
    return frames_dict


def parse_json_frames_joints(frames_json):
    frames_dict = {}
    for frame_comp in frames_json:
        frame_index = frame_comp["frame_index"]
        identity_bounding_boxes = frame_comp["joints"]
        dict_identities = {}
        for identity in identity_bounding_boxes:
            array_coordinates = [0] * 2
            array_coordinates[X_ID] = identity['joint']['x']
            array_coordinates[Y_ID] = identity['joint']['y']
            if identity["identity"] in dict_identities:
                print("ERRROR")
            dict_identities[identity["identity"]] = array_coordinates
        frames_dict[frame_index] = dict_identities
    return frames_dict


def interpolate_vals(first, second, mid_prop):
    return first + (second - first) * mid_prop


def interpolate_frame_bb(frame_id, cur_frame_id, next_frame_id, frames_bb_dict, bb_id):
    array_vals = [0.0] * 4
    proportion = (frame_id - cur_frame_id) / (next_frame_id - cur_frame_id)
    bound_box_cur = frames_bb_dict[cur_frame_id][bb_id]
    bound_box_next = frames_bb_dict[next_frame_id][bb_id]
    array_vals[X_ID] = interpolate_vals(bound_box_cur[X_ID],
                                        bound_box_next[X_ID],
                                        proportion)
    array_vals[Y_ID] = interpolate_vals(bound_box_cur[Y_ID],
                                        bound_box_next[Y_ID],
                                        proportion)
    array_vals[WEIGHT_ID] = interpolate_vals(bound_box_cur[WEIGHT_ID],
                                            bound_box_next[WEIGHT_ID],
                                            proportion)
    array_vals[HEIGHT_ID] = interpolate_vals(bound_box_cur[HEIGHT_ID],
                                            bound_box_next[HEIGHT_ID],
                                            proportion)
    if not frame_id in frames_bb_dict:
        frames_bb_dict[frame_id] = {}
    frames_bb_dict[frame_id][bb_id] = array_vals


def interpolate_frame_joint(frame_id, cur_frame_id, next_frame_id, frames_joint_dict, joint_id):
    array_vals = [0.0] * 2
    proportion = (frame_id - cur_frame_id) / (next_frame_id - cur_frame_id)
    joint_cur = frames_joint_dict[cur_frame_id][joint_id]
    joint_next = frames_joint_dict[next_frame_id][joint_id]
    array_vals[X_ID] = interpolate_vals(joint_cur[X_ID],
                                        joint_next[X_ID],
                                        proportion)
    array_vals[Y_ID] = interpolate_vals(joint_cur[Y_ID],
                                        joint_next[Y_ID],
                                        proportion)
    if not frame_id in frames_joint_dict:
        frames_joint_dict[frame_id] = {}
    frames_joint_dict[frame_id][joint_id] = array_vals


def interpolate_frame_seq(cur_frame_id, next_frame_id, frames_dict, object_id, is_bb: bool):
    for frame_id in range(cur_frame_id + 1, next_frame_id):
        if is_bb:
            interpolate_frame_bb(frame_id, cur_frame_id, next_frame_id, frames_dict, object_id)
        else:
            interpolate_frame_joint(frame_id, cur_frame_id, next_frame_id, frames_dict, object_id)


# Returns len of frame_ids in the there is not frame
def find_next_frame_with_id(frame_ids, bb_id, cur_frame_idx: int, frames_bb_dict: dict):
    next_frame_idx = frame_ids.__len__()
    for frame_idx in range(cur_frame_idx+1, frame_ids.__len__()):
        frame_id = frame_ids[frame_idx]
        if frame_id in frames_bb_dict and bb_id in frames_bb_dict[frame_id]:
            next_frame_idx = frame_idx
            return next_frame_idx
    return next_frame_idx


def interpolate_frames(frames_bb_dict: dict, frames_joints_dict: dict, bb_ids, joint_ids, bb_frame_ids, joint_frame_ids):
    for bb_id in bb_ids:
        # If bb_id exists there have to be at least one of them, thsu while will evaluate at least once
        cur_frame_idx = -1
        cur_frame_idx = find_next_frame_with_id(bb_frame_ids, bb_id, cur_frame_idx, frames_bb_dict)
        while cur_frame_idx < bb_frame_ids.__len__():
            next_frame_idx = find_next_frame_with_id(bb_frame_ids, bb_id, cur_frame_idx, frames_bb_dict)
            if (next_frame_idx >= bb_frame_ids.__len__()):
                break
            cur_frame_id = bb_frame_ids[cur_frame_idx]
            next_frame_id = bb_frame_ids[next_frame_idx]
            interpolate_frame_seq(cur_frame_id, next_frame_id, frames_bb_dict, bb_id, True)
            cur_frame_idx = next_frame_idx

    for joint_id in joint_ids:
        # If bb_id exists there have to be at least one of them, thsu while will evaluate at least once
        cur_frame_idx = -1
        cur_frame_idx = find_next_frame_with_id(joint_frame_ids, joint_id, cur_frame_idx, frames_joints_dict)
        while cur_frame_idx < joint_frame_ids.__len__():
            tmp_frame_idx = find_next_frame_with_id(joint_frame_ids, joint_id, cur_frame_idx, frames_joints_dict)
            if (tmp_frame_idx >= joint_frame_ids.__len__()):
                break
            next_frame_idx = tmp_frame_idx
            cur_frame_id = joint_frame_ids[cur_frame_idx]
            next_frame_id = joint_frame_ids[next_frame_idx]
            interpolate_frame_seq(cur_frame_id, next_frame_id, frames_joints_dict, joint_id, False)
            cur_frame_idx = next_frame_idx
        #TODO: Try side interpolation
        cur_frame_id = joint_frame_ids[cur_frame_idx]
        next_frame_id = joint_frame_ids[next_frame_idx]
        interpolate_frame_seq(cur_frame_id, next_frame_id, frames_joints_dict, joint_id, False)
        cur_frame_idx = next_frame_idx

def is_joint_inside_bb(coordinate, bound_box):
    if (coordinate[X_ID] >= bound_box[X_ID] and coordinate[X_ID] <= bound_box[X_ID] + bound_box[WEIGHT_ID]) and \
            (coordinate[Y_ID] >= bound_box[Y_ID] and coordinate[Y_ID] <= bound_box[X_ID] + bound_box[HEIGHT_ID]):
        return True
    return False


def get_all_identities(frames_dict: dict):
    identities = set()
    for frame_idx, vals in frames_dict.items():
        for identity in vals.keys():
            identities.add(identity)
    return identities


def get_all_frames_ids(frames_dict: dict):
    frames_ids = [int(key) for key in frames_dict.keys()]
    return sorted(frames_ids)


def get_joint_id_freq_max(joint_id_freq_dict: dict):
    freq_max = 0
    joint_id_max = -1
    for joint_id, freq in joint_id_freq_dict.items():
        if freq > freq_max:
            joint_id_max = joint_id
            freq_max = freq
    return joint_id_max, freq_max


def remove_empty(bb_id_potential_joint_id: dict):
    keys = [key for key in bb_id_potential_joint_id.keys()]
    for bb_id in keys:
        if bb_id_potential_joint_id[bb_id].__len__() == 0:
            del bb_id_potential_joint_id[bb_id]


def remove_joint_id(bb_id_potential_joint_id: dict, joint_id):
    for bb_id in bb_id_potential_joint_id:
        if joint_id in bb_id_potential_joint_id[bb_id]:
            del bb_id_potential_joint_id[bb_id][joint_id]
    remove_empty(bb_id_potential_joint_id)


def clean_unused_frames(frames_bb_dict: dict, frames_joints_dict: dict, bb_frame_ids, joints_frame_ids):
    min_frame_id = max(bb_frame_ids[0], joints_frame_ids[0])
    max_frame_id = min(bb_frame_ids[-1], joints_frame_ids[-1])
    bb_frame_ids_up = [key for key in frames_bb_dict.keys()]
    joints_frame_ids_up = [key for key in frames_joints_dict.keys()]
    for frame_id in bb_frame_ids_up:
        if (frame_id < min_frame_id) or (frame_id > max_frame_id):
            del frames_bb_dict[frame_id]
    for frame_id in joints_frame_ids_up:
        if (frame_id < min_frame_id) or (frame_id > max_frame_id):
            del frames_joints_dict[frame_id]


def map_joints_and_bbs(frames_bb_dict: dict, frames_joints_dict: dict):
    # Get ids, and frames
    bb_ids = get_all_identities(frames_bb_dict)
    joints_ids = get_all_identities(frames_joints_dict)
    bb_frame_ids = get_all_frames_ids(frames_bb_dict)
    joints_frame_ids = get_all_frames_ids(frames_joints_dict)

    interpolate_frames(frames_bb_dict, frames_joints_dict, bb_ids, joints_ids, bb_frame_ids, joints_frame_ids)
    #print(bb_frame_ids)
    #print(joints_frame_ids)

    bb_id_potential_joint_id = {k: dict() for k in bb_ids}
    clean_unused_frames(frames_bb_dict, frames_joints_dict, bb_frame_ids, joints_frame_ids)

    for frame_idx, vals in frames_bb_dict.items():
        #print(frame_idx, end=" ")
        for bb_id in bb_ids:
            if (bb_id not in vals):
                pass
                #print(frame_idx)
            else:
                if not frame_idx in frames_joints_dict:
                    pass
                    print("Hej")
                joints = frames_joints_dict[frame_idx]
                for joint_id, joint in joints.items():
                    if is_joint_inside_bb(joint, vals[bb_id]):
                        if joint_id in bb_id_potential_joint_id[bb_id]:
                            bb_id_potential_joint_id[bb_id][joint_id] += 1
                        else:
                            bb_id_potential_joint_id[bb_id][joint_id] = 1

                # TODO: Add cover check for other rectangles
                #print(vals["A"][HEIGHT_ID] * vals["A"][WEIGHT_ID], end=" ")
                #print(vals["A"][X_ID], vals["A"][Y_ID])

    print(bb_id_potential_joint_id)
    while not bb_id_potential_joint_id.__len__() == 0:
        joint_gl_id_max = -1
        bb_id_max = -1
        freq_gl_max = 0
        for bb_id, joint_id_freq_dict in bb_id_potential_joint_id.items():

            joint_id_max, freq_max = get_joint_id_freq_max(joint_id_freq_dict)
            if freq_max > freq_gl_max:
                freq_gl_max = freq_max
                joint_gl_id_max = joint_id_max
                bb_id_max = bb_id

        print("{}:{}".format(joint_gl_id_max, bb_id_max))
        del bb_id_potential_joint_id[bb_id_max]
        remove_joint_id(bb_id_potential_joint_id, joint_gl_id_max)

    pass

def run_tests():
    input_folder_a = []
    for test_number in TEST_NUMBERS:
        input_folder_a.append(TEST_PATH + str(test_number))

    # Run test cases.
    for input_folder in input_folder_a:
        print("Test path:" + input_folder)
        run_test(os.path.join(input_folder, "bboxes.json"), os.path.join(input_folder, "joints.json"))


def run_test(input_path_bb: str, input_path_joint: str):
    bounding_boxes = read_json_data(input_path_bb)
    joints = read_json_data(input_path_joint)
    frames_bounding_boxes = bounding_boxes["frames"]
    frames_joints = joints["frames"]

    # Multiple parts of the same person in the same picture???
    frames_bb_dict = parse_json_frames_bb(frames_bounding_boxes)
    frames_joints_dict = parse_json_frames_joints(frames_joints)
    map_joints_and_bbs(frames_bb_dict, frames_joints_dict)


def submision_test():
    bb_path = input()
    joints_path = input()
    run_test(bb_path, joints_path)

if __name__ == "__main__":

    run_tests()
    #submision_test()


