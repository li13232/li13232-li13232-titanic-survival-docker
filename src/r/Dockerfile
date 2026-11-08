FROM rocker/r-ver:4.4.3

WORKDIR /app
COPY install_packages.R .
RUN Rscript --vanilla install_packages.R
COPY model.R .

ENTRYPOINT ["Rscript", "--vanilla", "/app/model.R"]
