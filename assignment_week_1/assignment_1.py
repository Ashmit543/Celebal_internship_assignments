def lowertriangular(n):
    print("Lower Triangular Pattern:")
    for i in range(1, n + 1):
        print("* " * i)
    print()

def uppertriangular(n):
    print("Upper Triangular Pattern:")
    for i in range(n, 0, -1):
        print("* " * i)
    print()

def pyramid(n):
    print("Pyramid Pattern:")
    for i in range(1, n + 1):
        spaces = ' ' * (n - i)
        stars = '* ' * i
        print(spaces + stars)
    print()

# Set the height of the patterns
rows = 5

lower_triangular(rows)
upper_triangular(rows)
pyramid(rows)
