def parse_training_set(lines):
    training_set = {}
    for line in lines:
        tmp_trace_info = {}
        tmp_trace_info['trace'] = []
        begin_of_step = line.find(' ') + 1
        trace_id = int(line[:begin_of_step])
        is_human = False
        while True:
            end_of_step = line.find(';', begin_of_step)
            if end_of_step == -1:
                begin_of_step += 1
                comma = line.find(',', begin_of_step)
                last_space = line.find(' ', begin_of_step)
                target = []
                target.append(float(line[begin_of_step:comma]))
                target.append(float(line[comma+1:last_space]))
                tmp_trace_info['target'] = target
                if int(line[last_space+1:]) == 1:
                    is_human = True
                break
            position = []
            comma = line.find(',', begin_of_step)
            position.append(float(line[begin_of_step:comma]))
            comma2 = line.find(',', comma+1)
            position.append(float(line[comma+1:comma2]))
            position.append(float(line[comma2+1:end_of_step]))
            tmp_trace_info['trace'].append(position)
            begin_of_step = end_of_step + 1
        tmp_trace_info['is_human'] = is_human
        training_set[trace_id] = tmp_trace_info
    return training_set