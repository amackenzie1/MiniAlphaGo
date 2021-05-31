def do_stuff():
    i = 0
    while i < 20:
        i += 1
        yield i 

for i in do_stuff():
    print(i)