FROM rstudio/r-base:3.5-focal

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y \
  libpng-dev qpdf \
  python3 libpython3-dev python-is-python3 python3-venv \
  && rm -rf /var/lib/apt/lists/*


RUN R -e '{ \
  options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 2L); \
  deps <- unlist(tools::package_dependencies("reticulate", which = "most")); \
  install.packages(c(deps, "remotes", "rcmdcheck", "rmarkdown")); \
}'

RUN python -m venv /.virtualenvs/r-reticulate
RUN /.virtualenvs/r-reticulate/bin/python -m pip install --upgrade --no-user 'pip' 'wheel' 'setuptools'
RUN /.virtualenvs/r-reticulate/bin/python -m pip install --upgrade --no-user "numpy" "docutils" "pandas" "scipy" "matplotlib" "ipython" "tabulate" "plotly" "psutil" "kaleido" "wrapt"
ENV RETICULATE_PYTHON=/.virtualenvs/r-reticulate/bin/python

ADD ./ /reticulate
# RUN R -e 'remotes::install_local("/reticulate", dependencies = TRUE)'
RUN R -e 'rcmdcheck::rcmdcheck("/reticulate", args = c("--no-manual", "--as-cran"))'

# docker build -t reticulate-r-3-5 -f tools/r-3-5.Dockerfile .
# docker run -it reticulate-r-3-5 R -q -e 'rcmdcheck::rcmdcheck("/reticulate", args = c("--no-manual", "--as-cran"))'