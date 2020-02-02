from PIL import Image


def log_writer(writer, logs, step):
    for key, val in logs.items():
        writer.add_scalar(key, val, step)
    return writer
