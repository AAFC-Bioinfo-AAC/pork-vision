import time

def time_program(start):
    """
    Used to time how long certain things took in an image.
    """
    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds 