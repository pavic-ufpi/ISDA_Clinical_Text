import re
import snorkel
from snorkel.labeling import LFAnalysis
from snorkel.utils import probs_to_preds
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function
from sklearn.model_selection import train_test_split
from snorkel.labeling import filter_unlabeled_dataframe
 
    
#################### Global variables ####################

ABSTAIN = -1
RECEITAS = 0
EXAMES = 2
NOTAS = 1
    
#################### Ground-truth Label Functions ####################

def label_lookup(x, receitas, label):
    if x.TEXTO in receitas.values():
        return label
    return ABSTAIN

def make_label_lf(y, label_dicts):
    label_dict = label_dicts[y]
    name = f"label_{y}"
    return LabelingFunction(name, f=label_lookup, resources=dict(receitas=label_dict, label=y),)

def dataset_manualmente_rotulado(data):
    labels_by_label = data.groupby("LABEL")
    label_dicts = {}
    for LABEL in labels_by_label.groups:
        label_df = labels_by_label.get_group(LABEL)[["TEXTO"]]
        label_dicts[LABEL] = dict(zip(label_df.index, label_df.TEXTO)) 
    label_lfs = [make_label_lf(LABEL, label_dicts) for LABEL in label_dicts]
    return label_lfs

#################### Keyword Label Functions ####################

def keyword_lookup(x, keywords, label):
    if any(word in str(x.TEXTO) for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

def keywords():
    keyword_rec = make_keyword_lf(keywords=['uso', 'formula', 'oral', 'topico', 'otologico', 'interno', 'comp', 'contínuo', 
                                        'tratamento', 'manipulado', 'manipular', 'aplicar', 'tomar', 'via', 'gotas',
                                        'horas', 'pingar', 'ao dia', 'mg', 'cp', 'cx', 'sabonete', 'unidade'], label=RECEITAS)
    keyword_exa = make_keyword_lf(keywords=['solicitacao', 'solicito', 'exame', 'tc', 'rx', 'us', 'teste', 'ecg', 
                                            'vectonistagmografia'], label=EXAMES)
    keyword_out = make_keyword_lf(keywords=['paciente', 'relatorio', 'encaminho', 'alimento', 'avaliacao', 'dor', 'laudo', 
                                            'declaracao', 'declaro', 'atestado', 'atesto', 'retirar', 'encaminhamento'], 
                                            label=NOTAS)
    keyword_lfs = [keyword_rec, keyword_exa, keyword_out]
    return keyword_lfs

#################### Regex Label Functions ####################

def regex():
    @labeling_function()   
    def regex_receitas_ind_uso_int_mg(x):
        return RECEITAS if re.search(r"^(uso oral [a-zA-z]* \d* mg)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_intmg(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* [a-zA-z]* \d*mg)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_int_comp(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* [a-zA-z]* \d* comp)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_int_fr(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* [a-zA-z]* \d* fr)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_int_int_fr(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* \d*. [a-zA-z]* \d* fr)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_int_int(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* \d* [a-zA-z]* \d*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_int_intmg(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* \d* [a-zA-z]* \d*mg)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_uso_gota(x):
        return RECEITAS if re.search(r"^(uso [a-zA-z]* [a-zA-z]* \d* gota)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_ng_vd(x):
        return RECEITAS if re.search(r"^([a-zA-Z]* \d*ng [a-zA-Z]* \d* vd)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_2_ds_cx_comp(x):
        return RECEITAS if re.search(r"(^\d*. [a-zA-Z]* \d* [a-zA-Z]* \d* cx [a-zA-Z]* \d comp)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()   
    def regex_receitas_ind_ds_cx_comp(x):
        return RECEITAS if re.search(r"(^\d*. [a-zA-Z]* \d* [a-zA-Z]* \d*cx [a-zA-Z]* \d comp)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_solicitacao(x):
        return EXAMES if re.search(r"^(solicitacao de exame: no. \d* convenio)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_solicito(x):
        return EXAMES if re.search(r"^(solicito [a-zA-Z]*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_grama(x):
        #programa e grama não se incluem
        return EXAMES if re.search(r"\b(?!grama|programa\b).*grama", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_terapia(x):
        return EXAMES if re.search(r"\b.*terapia", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_oscopia(x):
        return EXAMES if re.search(r"\b.*oscopia", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_exames_exames(x):
        return EXAMES if re.search(r"^(exames [a-zA-Z]*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_paciente(x):
        return NOTAS if re.search(r"^(paciente com.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_relatorio(x):
        return NOTAS if re.search(r"^(relatorio medico.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_encaminho(x):
        return NOTAS if re.search(r"^(encaminho a.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_avaliacao(x):
        return NOTAS if re.search(r"^(avaliacao.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_solicito(x):
        return NOTAS if re.search(r"^(solicito encaminhamento.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    @labeling_function()
    def regex_outros_alimentos(x):
        return NOTAS if re.search(r"^(alimentos bons para o.*)", str(x.TEXTO), flags=re.I) else ABSTAIN

    regex_lfs = [regex_exames_grama, regex_exames_terapia, regex_exames_oscopia, 
           regex_exames_solicitacao, regex_exames_solicito, regex_exames_exames, regex_receitas_ind_uso_int_mg, 
           regex_receitas_ind_uso_intmg, regex_receitas_ind_uso_int_comp, regex_receitas_ind_uso_int_fr, 
           regex_receitas_ind_uso_int_int_fr, regex_receitas_ind_uso_int_int, regex_receitas_ind_uso_int_intmg, 
           regex_receitas_ind_uso_gota, regex_receitas_ind_ng_vd, regex_receitas_ind_2_ds_cx_comp, 
           regex_receitas_ind_ds_cx_comp, regex_outros_paciente, regex_outros_relatorio, regex_outros_encaminho, 
           regex_outros_avaliacao, regex_outros_solicito, regex_outros_alimentos]

    return regex_lfs


#################### Snorkel ####################

def snorkel(data):
    
    keyword_lfs = keywords()
    regex_lfs = regex()
    
    size = len(data)/2
    data_label = data[:int(size)]
    label_lfs = dataset_manualmente_rotulado(data_label)
    
    lfs = keyword_lfs + regex_lfs + label_lfs
    
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=data)
    
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)
    probs_train = label_model.predict_proba(L=L_train)

    #X, y = filter_unlabeled_dataframe(X=data.RECEITA, y=probs_train, L=L_train)
    #y = probs_to_preds(probs=y)
    return data.TEXTO, probs_to_preds(probs_train)
    
    