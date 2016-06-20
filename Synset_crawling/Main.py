import FileIO
import Node
import numpy as np

def main():

    synsets = FileIO.read_csv('synsets.csv')
    words = FileIO.read_csv('words.txt', '\t')
    is_a = FileIO.read_csv('wordnet.is_a.txt')

    data = {}
    # for parent,child in is_a:
    #     if child not in data:
    #         c = Node.Node()
    #         c.id = child
    #         c.parent = parent
    #         data[child] = c

    # fill dict with synsets (Node objects)
    for word in words:
        cl = word[0]
        if cl not in data:
            w = Node.Node()
            w.name = cl
            w.children = []
            data[cl] = w

    # find parent and children for every node
    for parent,child in is_a:
        n = data[parent]
        n.children.append(child)

    # check our synsets with this 'tree' structure and find all children + grandchildren
    i = 0
    sum = 0
    for synset in synsets:
        n = data[synset]
        n.id = i
        subtree = []
        if len(n.children) > 0:
            subtree.append(n.children)
            for child in n.children:
                n2 = data[child]
                if len(n2.children) > 0:
                    subtree.append(n2.children)

            n.children = subtree[0]

        i += 1

        sum += len(n.children)
    print sum
        # print n.toString()

    # write for wget download
    with open('new_synsets.txt', 'w') as f:
        for synset in synsets:
            n = data[synset]
            for child in n.children:
                f.write(child + "\n")
    f.close()

    # write with labels
    with open('new_synsets_with_labels.txt', 'w') as f:
        for synset in synsets:
            n = data[synset]
            for child in n.children:
                f.write(child + "," + str(n.id) + "\n")
    f.close()

    # write with parent class
    with open('copy_command.sh', 'w') as f:
        f.write("#!/bin/sh\n")
        for synset in synsets:
            n = data[synset]
            for child in n.children:
                f.write("cp " + "./synsets/" + child + "/. " + "./synsets_all/" + synset + "/ -R" + "\n")
    f.close()


    print "a"

if __name__ == '__main__':
    main()