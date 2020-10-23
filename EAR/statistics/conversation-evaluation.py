import regex as re
import numpy as np
from collections import defaultdict

fp_list = []


def eval_lastfm(fp_list):
    print('Evaluating LastFM')
    dir = '../lastfm/data/interaction-log/ear'
    f1 = 'v4-code-stable-s-0-e-20000-lr-0.001-gamma-0.0-playby-policy-stra-maxent-topK-3-trick-0-eval-1-init-0-mini-1-always-1-upcount-4-upreg-0.001-m-0.txt'
    f2 = 'v5-code-stable-s-0-e-2000-lr-0.001-gamma-0.7-playby-sac-stra-maxent-topK-3-trick-0-eval-0-init-0-mini-1-always-1-upcount-1-upreg-0.001-m-0.txt'
    f3 = 'v5-code-stable-s-0-e-11800-lr-0.001-gamma-0.7-playby-sac-stra-maxent-topK-3-trick-0-eval-0-init-0-mini-1-always-1-upcount-1-upreg-0.001-m-0.txt'
    fp_list.append('{}/{}'.format(dir, f1))
    fp_list.append('{}/{}'.format(dir, f2))
    fp_list.append('{}/{}'.format(dir, f3))


    dir2 = '../lastfm/data/interaction-log/crm'
    f4 = 'v4-code-stable-s-0-e-20000-lr-0.001-gamma-0.7-playby-policy-stra-maxent-topK-3-trick-0-eval-1-init-0-mini-0-always-0-upcount-0-upreg-0.001-m-0.txt'
    fp_list.append('{}/{}'.format(dir2, f4))
    #fp_list.append('{}/{}'.format(dir, f5))
    #fp_list.append('{}/{}'.format(dir, f6))
    #fp_list.append('{}/{}'.format(dir, f7))
    #fp_list.append('{}/{}'.format(dir, f8))
    return


def eval_yelp(fp_list):
    print('Evaluating Yelp')
    dir = '../yelp/data/interaction-log/ear'
    f1 = 'v5-code-stable-s-0-e-1000-lr-0.001-gamma-0.7-playby-sac-stra-maxent-topK-3-trick-0-eval-0-init-0-mini-1-always-1-upcount-1-upreg-0.001-m-0.txt'
    f2 = 'v4-code-stable-s-0-e-20000-lr-0.001-gamma-0.7-playby-policy-stra-maxent-topK-3-trick-0-eval-1-init-0-mini-1-always-1-upcount-1-upreg-0.001-m-0.txt'
    fp_list.append('{}/{}'.format(dir, f1))
    fp_list.append('{}/{}'.format(dir, f2))

    dir2 = '../yelp/data/interaction-log/crm'
    f3 = 'v4-code-stable-s-0-e-20000-lr-0.001-gamma-0.0-playby-policy-stra-maxent-topK-3-trick-0-eval-1-init-0-mini-1-always-1-upcount-4-upreg-0.001-m-0.txt'
    fp_list.append('{}/{}'.format(dir2, f3))
    return


def segment_list(l):
    buffers = []
    buffer = []
    for i in range(len(l)):
        if len(buffer) == 0:
            buffer.append(l[i])

        if i < len(l) - 1 and l[i + 1] - 1 == l[i]:
            buffer.append(l[i + 1])
        else:
            buffers.append(buffer)
            buffer = []
    return buffers


def parse(episodes, lth, rth, turncnt):
    turn_count_dict = defaultdict(list)
    rec_success_list = list()

    attempt_rec_length = list()
    count = 0

    ent_dict = defaultdict(list)
    can_dict = defaultdict(list)

    epi_count = 0
    for episode in episodes:
        epi_count += 1
        # if epi_count > 10000:
        #     continue
        ent_list = []
        rec_turns = []
        lines = episode.split('\n')
        if len(lines) < 3:
            continue

        candidate_length_list = list()
        in_th = False
        for line in lines:
            if re.search(r'episode count', line):
                nums = len(re.findall(r'[0-9]+', line))
                fea_count = nums - 3

                if fea_count >= lth and fea_count <= rth:
                    in_th = True

        if in_th is False:
            continue
        else:
            count += 1

        rej_cnt = 0
        for line in lines:
            if re.search(r'candidate length', line):
                turn_count = re.findall(r'[0-9]+', line)[0]
                length = re.findall(r'[0-9]+', line)[1]
                turn_count_dict[int(turn_count)].append(int(length))
                candidate_length_list.append(int(length))

            if re.search(r'ACCEPT_REC', line):
                turn_count = re.findall(r'[0-9]+', line)[0]
                rec_turns.append(int(turn_count))

                target_position = re.findall(r'[0-9]+', line)[-2]
                num_rec = re.findall(r'[0-9]+', line)[-1]
                rec_success_list.append((int(target_position), int(num_rec)))

                attempt_rec_length.append(candidate_length_list[-1])


            if re.search(r'REJECT_REC', line):
                turn_count = re.findall(r'[0-9]+', line)[0]
                rec_turns.append(int(turn_count))

                rej_cnt += 1

            if re.search(r'Ent Sum', line):
                if len(re.findall(r'[0-9]+\.[0-9]+', line)) > 0:
                    ent = float(re.findall(r'[0-9]+\.[0-9]+', line)[0])
                else:
                    ent = 0
                ent_list.append(ent)

            turn_count_dict_len = {}
            for k, v in turn_count_dict.items():
                turn_count_dict_len[k] = len(v)

        buffers = segment_list(rec_turns)

        if len(buffers) > 0 and len(buffers[-1]) == 1:
            buffers = buffers[: -1]

    remain_list = []
    success_list = []
    for i in range(turncnt):
        remain_list.append(count + 1 - len(turn_count_dict[i]))
        success_list.append(1 - len(turn_count_dict[i]) / float(count + 1))

    print('total epi: {}'.format(count))

    for index, item in enumerate(success_list):
        print('turn, {}, SR, {}'.format(index + 1, '{0:.3f}'.format(item)))

    for key in sorted(ent_dict.keys()):
        print('rec length: {}, ent: {}, median: {}, support: {}'.format(key, np.mean(np.array(ent_dict[key])), np.median(np.array(ent_dict[key])), len(ent_dict[key])))

    for key in sorted(can_dict.keys()):
        print('rec length: {}, ent: {}, median: {}, support: {}'.format(key, np.mean(np.array(can_dict[key])), np.median(np.array(can_dict[key])), len(can_dict[key])))
    s = 0
    num = 0
    for i in range(len(remain_list) - 1):
        s += (remain_list[i + 1] - remain_list[i]) * (i + 1)
        num += remain_list[i + 1] - remain_list[i]

    s += (count - remain_list[-1]) * turncnt
    num += (count - remain_list[-1])

    print('average: {0:.3f}'.format((float(s) / num)))

    success_rate = len(rec_success_list) / float(len(episodes))

    rec = list()
    for item in rec_success_list:
        a = item[0] / float(item[1])
        rec.append(a)

    print('success rate: {}, naive rec index: {}'.format(success_rate, np.mean(np.array(rec))))


if __name__ == '__main__':
    # choose what you want to evaluate
    #eval_lastfm(fp_list)
    eval_yelp(fp_list)

    for fp in fp_list:
        with open(fp, 'r') as f:
            lines = f.read()
        episodes2 = lines.split('Starting new\n')
        print('fp is: {}'.format(fp))
        parse(episodes2, 0, 33, 15)

    print('done')
