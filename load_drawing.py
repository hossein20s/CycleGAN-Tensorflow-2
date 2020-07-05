from quickdraw import QuickDrawData

if __name__ == '__main__':
    quickdraw = QuickDrawData()
    print(quickdraw.drawing_names)
    n = 10
    for i in range(n*2):
        draw = quickdraw.get_drawing("bicycle")
        draw.image.save("datasets/trainA/train_bicycle{}.gif".format(i))
    for i in range(n):
        draw = quickdraw.get_drawing("bicycle")
        draw.image.save("datasets/testA/test_bicycle{}.gif".format(i))
    for i in range(n*2):
        draw = quickdraw.get_drawing("campfire")
        draw.image.save("datasets/trainB/train_campfire{}.gif".format(i))
    for i in range(n):
        draw = quickdraw.get_drawing("campfire")
        draw.image.save("datasets/testB/test_campfire{}.gif".format(i))
