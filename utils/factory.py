def get_model(model_name, args):
    name = model_name.lower()
    if name=='engine':
        from models.engine import Learner
        return Learner(args)
    else:
        assert 0
