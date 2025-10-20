import os
import sys
import torch
import logging
import torch.nn as nn
def loadModel(hyperparams, netname, channels):
    Netpath = 'Model'
    Netfile = os.path.join(Netpath, netname)
    Netfile = os.path.join(Netfile, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'resnet50': 
        return loadresnet50(hyperparams, channels)

    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))

        sys.exit()

def get_model_list(netname = ''):
    netname = netname.lower()
    if netname == '':
        return ['resnet50']

    if netname in ['resnet50']:
        return [netname]

    logging.warning("No model with the name {} found, please check your spelling.".format(netname))
    logging.warning("Net List:")
    logging.warning("    resnet50")
    sys.exit()
    
  
def loadresnet50(hyperparams, channels):
    from Model.resnet50.resnet50 import resnet50
    logging.info("Loading resnet50 Model")
    return resnet50(hyperparams, channels)

    
