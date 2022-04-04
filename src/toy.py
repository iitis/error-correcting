from src.DIRAC import DIRAC
from src.environment import Chimera
from src.data_gen import generate_chimera

device = 'cuda:0'
model = DIRAC()
model = model.to(device)
env = Chimera(generate_chimera(1, 1))



