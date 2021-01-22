if __name__ == '__main__':
    students = []
    scores = []
    final_names = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        scores.append(score)
        students.append([name, score])

    scores = sorted(list(set(scores)))
    second_low = scores[1]

    for student in students:
        if student[1] == second_low:
            final_names.append(student[0])

    final_names.sort()
    for i in final_names:
        print(i)





