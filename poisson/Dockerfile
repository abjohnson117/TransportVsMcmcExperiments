FROM quay.io/fenicsproject/dev

# ---------- Python deps (installed system-wide) ----------
USER root
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
        hippylib \
        jupyter \
        matplotlib \
        h5py \
        pyyaml \
        seaborn==0.10.0 \
        statsmodels \
        aiohttp \
        requests

# ---------- MUQ install ----------
WORKDIR /home/fenics
RUN apt update && apt install -y --no-install-recommends \
        libboost-all-dev \
        libhdf5-dev \
        git \
        cmake \
        build-essential && \
    mkdir -p /home/fenics/lib/muq && \
    rm -rf /home/fenics/muq2 && \
    git clone --depth 1 https://bitbucket.org/mituq/muq2.git && \
    cd muq2 && mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq \
          -DMUQ_USE_MPI=OFF \
          -DMUQ_USE_PYTHON=ON .. && \
    make -j"$(nproc)" install && \
    cd /home/fenics && rm -rf muq2


# ---------- hippylib–MUQ interface ----------
USER fenics
RUN git clone https://github.com/hippylib/hippylib2muq.git /home/fenics/hippylib2muq

# ---------- Environment for MUQ + hippylib2muq ----------
ENV LD_LIBRARY_PATH="/home/fenics/lib/muq/lib:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/home/fenics/lib/muq/python:/home/fenics/hippylib2muq:${PYTHONPATH}"

# We will mount your repo at runtime here:
WORKDIR /home/fenics/poisson

# Default to a shell (phusion's my_init will eventually exec bash)
CMD ["bash"]









# FROM quay.io/fenicsproject/dev

# USER root
# RUN pip3 install --upgrade pip

# USER fenics

# CMD ["bash"]

# RUN pip3 install hippylib && \
#     pip3 install jupyter && \
#     pip3 install matplotlib && \
#     pip3 install h5py && \
#     pip3 install pyyaml && \
#     pip3 install seaborn==0.10.0 && \
#     pip3 install statsmodels && \
#     pip3 install aiohttp && \
#     pip3 install requests

# WORKDIR /home/fenics/

# # Install MUQ
# USER root
# RUN apt update && apt install -y libboost-all-dev libhdf5-dev && \
#     mkdir -p ./lib/muq && \
#     git clone --depth 1 https://bitbucket.org/mituq/muq2.git && \
#     cd muq2/; mkdir build; cd build && \
#     cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq -DMUQ_USE_MPI=OFF -DMUQ_USE_PYTHON=ON ../ && \
#     make -j24 install && \
#     cd /home/fenics/ && \
#     rm -rf muq2
# USER fenics

# # Install hippylib-muq interface
# RUN git clone https://github.com/hippylib/hippylib2muq.git

# # Set environmental variables
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/fenics/lib/muq/lib" \
#     PYTHONPATH="/home/fenics/lib/muq/python:/home/fenics/hippylib2muq"

# COPY poisson.py poisson.yaml /home/fenics/app/

# USER root

# WORKDIR /home/fenics/app/
# CMD ["python3 /home/fenics/app/poisson.py"]

# FROM quay.io/fenicsproject/dev

# # ---- base setup ----
# USER root
# RUN pip3 install --upgrade pip

# # Python libs (installed as fenics user)
# USER fenics
# RUN pip3 install \
#     hippylib \
#     jupyter \
#     matplotlib \
#     h5py \
#     pyyaml \
#     seaborn==0.10.0 \
#     statsmodels \
#     aiohttp \
#     requests

# # ---- MUQ build (needs system libs) ----
# USER root
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       libboost-all-dev libhdf5-dev git cmake build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Install MUQ (prefix into /home/fenics/lib/muq)
# RUN mkdir -p /home/fenics/lib/muq && \
#     cd /home/fenics && \
#     git clone --depth 1 https://bitbucket.org/mituq/muq2.git && \
#     cd muq2 && mkdir build && cd build && \
#     cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq \
#           -DMUQ_USE_MPI=OFF -DMUQ_USE_PYTHON=ON ../ && \
#     make -j8 install && \
#     cd /home/fenics && rm -rf muq2

# # hippylib <-> muq interface
# USER fenics
# RUN git clone https://github.com/hippylib/hippylib2muq.git /home/fenics/hippylib2muq

# # env
# ENV LD_LIBRARY_PATH="/home/fenics/lib/muq/lib:${LD_LIBRARY_PATH}" \
#     PYTHONPATH="/home/fenics/lib/muq/python:/home/fenics/hippylib2muq:${PYTHONPATH}"

# # Work in this path; at runtime we'll mount your repo here
# WORKDIR /home/fenics/poisson

# # Default to a shell (no auto-run)
# CMD ["bash"]

# FROM quay.io/fenicsproject/dev

# # Upgrade pip under root
# USER root
# RUN pip3 install --upgrade pip

# # Back to fenics for Python packages
# USER fenics
# RUN pip3 install \
#     hippylib \
#     jupyter \
#     matplotlib \
#     h5py \
#     pyyaml \
#     seaborn==0.10.0 \
#     statsmodels \
#     aiohttp \
#     requests

# # ---------- Install MUQ ----------
# USER root
# WORKDIR /home/fenics/
# RUN apt update && apt install -y libboost-all-dev libhdf5-dev && \
#     mkdir -p ./lib/muq && \
#     rm -rf muq2 && \
#     git clone --depth 1 https://bitbucket.org/mituq/muq2.git && \
#     cd muq2 && mkdir -p build && cd build && \
#     cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/lib/muq -DMUQ_USE_MPI=OFF -DMUQ_USE_PYTHON=ON .. && \
#     make -j"$(nproc)" install && \
#     cd /home/fenics/ && \
#     rm -rf muq2

# # hippylib → MUQ interface
# USER fenics
# RUN git clone https://github.com/hippylib/hippylib2muq.git

# # MUQ + hippylib2muq environment
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/fenics/lib/muq/lib" \
#     PYTHONPATH="/home/fenics/lib/muq/python:/home/fenics/hippylib2muq"

# # ---------- Copy your Poisson code ----------
# # IMPORTANT: This Dockerfile must be located in the same directory as these four files.
# WORKDIR /home/fenics/app/
# COPY poisson.py poisson.yaml qoi_tracing.py generate_training_dataset.py /home/fenics/app/

# # Ensure runtime user is fenics
# USER fenics
