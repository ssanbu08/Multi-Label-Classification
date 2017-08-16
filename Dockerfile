FROM continuumio/anaconda3
MAINTAINER Anbarasan ss.anbuselvan@gmail.com


# 1. RUN
RUN pip install scikit-multilearn
RUN pip install orange3-associate
#RUN pip install gensim
#RUN pip install pyLDAvis
#RUN /opt/conda/bin/conda install jupyter -y --quiet 
#RUN pip install simplejson
#RUN mkdir /opt/notebooks





# 2. Copy the Source code
COPY . /src

#3.CMD
CMD ['top']



# 3. Set the Entrypoint
# ENTRYPOINT [ “/bin/bash”, “-c” ]