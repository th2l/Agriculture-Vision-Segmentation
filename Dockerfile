FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
RUN pip install --no-cache-dir tabulate tensorflow_probability pandas matplotlib Pillow -i http://ftp.daumkakao.com/pypi/simple --trusted-host=ftp.daumkakao.com
RUN pip install --no-cache-dir --upgrade tensorflow_addons -i http://ftp.daumkakao.com/pypi/simple --trusted-host=ftp.daumkakao.com
WORKDIR /mnt/SharedProject/
