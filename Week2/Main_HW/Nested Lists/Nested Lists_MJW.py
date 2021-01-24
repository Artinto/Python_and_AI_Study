if __name__ == '__main__':
    
    records = dict()  # key: name // value: score
    
    for _ in range(int(input())):
        name = input()
        score = float(input())
        
        records[name] = score
        
    scores = sorted(set(records.values()))[1]   # the second lowest grade 
    
    for k, v in sorted(records.items()):
        if v == scores:
            print(k)
