def size_filter(cells, smin, smax):
    for cell in cells:
        if smin < cell.area < smax:
            yield cell

def segmented_area(mask):
    return mask.astype(bool).sum() / mask.size

def artefact_filter():
    pass

def co_expression_intracell():
    pass

def co_expression_neighboor():
    pass