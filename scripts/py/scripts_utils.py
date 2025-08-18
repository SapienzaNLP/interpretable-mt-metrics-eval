def read_metric2latex(filepath='scripts/py/metric2latex.txt'):
    with open(filepath) as f:
        lines = f.readlines()

    metric2latex = {}
    for line in lines:
        line = line.strip().split(':')
        metric = line[0].strip()
        latex = line[1].strip()

        metric2latex[metric] = latex
    
    return metric2latex

def read_metric2command(filepath='scripts/py/metric2command.txt'):
    with open(filepath) as f:
        lines = f.readlines()

    metric2latex = {}
    for line in lines:
        line = line.strip().split(':')
        metric = line[0].strip()
        latex = line[1].strip()

        metric2latex[metric] = latex
    
    return metric2latex