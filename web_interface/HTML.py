import os.path

def html_stuff(filename1, filename2, output):
    data_prefix = 0
    data_vars2 = 'var all_scenes = ['
    data_suffix = 0

    path = './shot_images/all_shots'
    num_files = len([f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))])

    for i in range(1, num_files):
        pref = ''
        # if i / 100 == 0:
        #     pref += '0'
        # if i / 10 == 0:
        #     pref += '0'

        data_vars2 += pref + str(i) + ', '

    data_vars2 += '904' + '];\n'


    #read prefix and suffix (html and javascript code)
    with open(filename1, 'r') as f1:
        data_prefix = f1.read()
    f1.close()
    with open(filename2, 'r') as f2:
        data_suffix = f2.read()
    f2.close()

    string = data_prefix + data_vars2 + data_suffix

    with open(output, 'w') as f:
        f.write(string)
    f.close()
