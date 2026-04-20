from models.iTransformer import Model as iTransformer
from models.Transformer import Model as Transformer
from models.iInformer import Model as iInformer
from models.Informer import Model as Informer
from models.iReformer import Model as iReformer
from models.Reformer import Model as Reformer
from models.iFlowformer import Model as iFlowformer
from models.Flowformer import Model as Flowformer
from models.iFlashformer import Model as iFlashformer
from models.Flashformer import Model as Flashformer

model_dict = {
    'iTransformer': iTransformer,
    'Transformer': Transformer,
    'iInformer': iInformer,
    'Informer': Informer,
    'iReformer': iReformer,
    'Reformer': Reformer,
    'iFlowformer': iFlowformer,
    'Flowformer': Flowformer,
    'iFlashformer': iFlashformer,
    'Flashformer': Flashformer,
}
