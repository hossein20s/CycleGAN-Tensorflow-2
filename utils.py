import time

import PIL
import tensorflow
from matplotlib import pyplot, animation

cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tensorflow.ones_like(fake_output), fake_output)

# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

def make_animation(image_buffers, anomation_file_name):
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 10))
    pyplot.close()
    ax.xlim = (0, 10)
    ax.ylim = (0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    buffer = image_buffers[0]
    buffer.seek(0)
    img = ax.imshow(pyplot.imread(buffer))
    img.set_interpolation('nearest')

    def updatefig(frame, buffers):
        buffer = buffers[frame]
        buffer.seek(0)
        img.set_data(pyplot.imread(buffer))
        return (img,)

    animation_function = animation.FuncAnimation(fig, updatefig, frames=len(image_buffers), fargs=(image_buffers,),
                                                 interval=100, blit=True)
    anim_writer = animation.writers['ffmpeg']
    writer = anim_writer(fps=15, metadata=dict(artist='Hossein'), bitrate=1800)
    time_string = time.strftime("%Y%m%d-%H%M%S")
    file_name = '{}-{}.mp4'.format(anomation_file_name, time_string)
    animation_function.save(file_name, writer=writer, )
    return image_buffers
