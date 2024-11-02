from preprocessing import *
from training import *

if __name__ == '__main__':
    build_artefacts()
    join_artefacts()
    train_model()
    validate_model()

# Reference: https://medium.com/@lixg2000/conditional-random-fields-c7a872f22780
# Reference: https://dev.to/fferegrino/conditional-random-fields-in-python-sequence-labelling-part-4-5ei2
