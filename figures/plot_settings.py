import matplotlib as mpl
import matplotlib.pyplot as plt


# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42

MEDIUM_SIZE = 10
SMALLER_SIZE = 8
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('font', family='Helvetica')
FIG_HEIGHT = 4
FIG_WIDTH = 4


MODELS = [
    "gpt2", "bert-base-uncased", "bert-large-uncased", "facebook/bart-base", "facebook/bart-large",
    "roberta-base", "roberta-large", "microsoft/phi-2", "google/gemma-2b", "google/gemma-7b",
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"
]


def get_square_axis():
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_wider_axis(double=False):
    plt.figure(figsize=(int(FIG_WIDTH * (3/2 if not double else 5/2)), FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_double_square_axis():
    plt.figure(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
    return ax


def get_model_ordering(actual_models):
    return sorted(actual_models, key=lambda m: MODELS.index(m))


def get_metric_name(name):
    return {
        "economic_score": "economic axis",
        "social_score": "social axis",
    }[name]


def get_model_family(mod):
    details = mod.split("/")[-1]
    return details.split("-")[0] if details != "gpt2" else "gpt"


def get_model_class_size(mod):
    family = get_model_family(mod).lower()
    size_info = mod.split("/")[-1].split("-")
    for sz in size_info:  # one of these contains info
        if sz == "base":
            return sz
        elif sz == "large":
            return sz
        elif len(sz) == 2 and "b" == sz[1]:
            if family == "llama" and sz == "7b":  # standard
                return "standard"
            elif family == "gemma":
                return "large" if sz == "7b" else "base"
    return "standard"


def get_model_training_type(mod):
    if "chat" in mod or "it" in mod:
        return "chat"
    return "pretrained"


def get_model_colors(mod):
    return {
        'gpt': '#1b9e77',
        'bert': '#d95f02',
        'bart': '#7570b3',
        'roberta': '#66a61e',
        'phi': '#e6ab02',
        'gemma': '#a6761d',
        'llama': '#e7298a'
    }[get_model_family(mod).lower()]


def get_model_marker(mod):
    return {
        "standard": "o",
        "base": "o",
        "large": "P",
    }[get_model_class_size(mod)]


def get_model_edgecolor(mod):
    return {
        "chat": "#000000",
        "pretrained": "#808080",
    }[get_model_training_type(mod)]


def get_sag_vs_baseline_colors(mod):
    if mod in {'Sagittarius'}:
        return '#c7eae5'
    else:
        return '#dfc27d'


def get_model_name_conventions(mname):
    return {
        "gpt2": "GPT2",
        "bert-base-uncased": "BERT (base)",
        "bert-large-uncased": "BERT (large)",
        "facebook/bart-base": "BART (base)",
        "facebook/bart-large": "BART (large)",
        "roberta-base": "RoBERTa (base)",
        "roberta-large": "RoBERTa (large)",
        "microsoft/phi-2": "Phi-2",
        "google/gemma-2b": "Gemma (2b)",
        "google/gemma-7b": "Gemma (7b)",
        "meta-llama/Llama-2-7b-hf": "Llama (7b)",
        "meta-llama/Llama-2-7b-chat-hf": "Llama chat (7b)"
    }[mname]
    
    
def get_LINCS_task_names(task, add_return=True):
    if task == 'continuousCombinatorialGeneration':
        return 'Complete{}Generation'.format('\n' if add_return else ' ')
    elif task == 'comboAndDosage':
        return 'Combination{}& Dosage'.format('\n' if add_return else ' ')
    elif task == 'comboAndTime':
        return 'Combination{}& Treatment Time'.format('\n' if add_return else ' ')
    print('Unrecognized', task)
    assert False
    
    
def get_species_axis_tick(species):
    if species == 'RhesusMacaque':
        return 'Rhesus\nMacaque'  # make this two lines
    return species


def get_organ_color_palette():
    return ['#01665e', '#5ab4ac', '#c7eae5', '#f6e8c3', '#dfc27d', '#bf812d', '#8c510a']


def get_TM_color_palette():
    return {
        'Sagittarius': '#80cdc1',
        'edge': '#003c30',
        'baseline': '#018571',
        'Heart': '#f5f5f5',
        'Kidney': '#dfc27d',
        'Liver': '#a6611a' }


def get_base_color():
    return '#dfc27d'


def get_line_style(mod):
    if mod in {'Sagittarius'}:
        return 'solid'
    elif mod == 'mean':
        return (0, (5, 1))
    elif mod == 'linear':
        return 'dotted'
    elif mod == 'seq_by_seq_neuralODE':
        return 'dotted'
    elif mod == 'seq_by_seq_RNN':
        return (0, (5, 1))
    elif mod == 'Sagittarius_ablation_noCensored':
        return 'dotted'
    elif mod == 'Sagittarius_ablation_allCensored':
        return (0, (5, 1))
    raise ValueError("Unknown model: {}".format(mod))