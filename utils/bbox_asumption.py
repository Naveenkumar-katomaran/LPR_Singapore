def list_of_points(pt1, pt2, length):
    a = (pt2[0] - pt1[0]) / length
    b = (pt2[1] - pt1[1]) / length
    ab = [pt1]
    at = pt1[0]
    bt = pt1[1]
    for _ in range(length - 2):
        at = at + a
        bt = bt + b
        ab.append((round(at), round(bt)))
    ab.append(pt2)
    # print(ab)
    return ab


def rect_points(list1, length):
    if len(list1) == 1:
        return list1

    rect_list = list()
    for i in range(0, len(list1) - 1):
        start = list_of_points(list1[i][0], list1[i + 1][0], length)
        end = list_of_points(list1[i][1], list1[i + 1][1], length)
        for j, k in zip(start, end):
            rect_list.append([(int(j[0]), int(j[1])), (int(k[0]), int(k[1]))])
    return rect_list
