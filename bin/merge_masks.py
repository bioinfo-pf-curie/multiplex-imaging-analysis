
## test sur les flows non concluant ##

# import numpy as np

# def load_npy(npy_path):
#     """Helper to load npy files"""
#     return np.load(npy_path, allow_pickle=True).item()['flows'][4]

# c = load_npy("test_normalize_nuc.ome_seg_cyto.npy")
# tn = load_npy("test_normalize_nuc.ome_seg_tn.npy")

# out = np.lib.format.open_memmap("result.npy", dtype="float32", shape=c[4].shape, mode="w+")
# out[...] = np.max([c[4], tn[4]], axis=0)
# out.flush()


## pseudo code pour merge masks ##

# objs = find_object(mask_cyto) 
# autre = find_object(mask_tn)
# pour chaque cell de objs:
#   si cell partage sa zone avec une dans autre (et seulement une) alors
#       si IoU > 70 (pe changer ça) alors suprimer dans autre
#        sinon nouvel cell (union) + supr les deux
#   si partage avec plusieurs alors 
#       si grande diff d'aire alors union avec la + grosse et intersection avec la plus petite commune
#       sinon on fusionne tout
#   si pas de cell correspondante alors on enregistre la cell
# pour chaque cell restante de autre:
#   enregistrer dans le resultat