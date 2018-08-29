def update_weights(dst_model, src_model, alpha):
    for dst_p, src_p in zip(dst_model.parameters(), src_model.parameters()):
        dst_p.data.mul_(alpha).add_(1 - alpha, src_p.data)


class SWAWeightsOptimizer(object):
    def __init__(self, model):
        self.model = model
        self.n_models = 0

    def getNModels(self):
        return self.n_models

    def setNModels(self, n_models):
        self.n_models = n_models

    def update(self, student_model):
        self.n_models += 1
        if self.n_models == 1:
            self.model.load_state_dict(student_model.state_dict())
        else:
            alpha = 1. - 1. / float(self.n_models)
            update_weights(self.model, student_model, alpha)


class EMAWeightsOptimizer(object):
    def __init__(self, model, ema_decay):
        self.model = model
        self.ema_decay = ema_decay

    def update(self, student_model, global_step):
        alpha = min(1 - 1 / (global_step + 1), self.ema_decay)
        update_weights(self.model, student_model, alpha)


class MeanWeightsOptimizer(object):
    def __init__(self, model):
        self.model = model

    def update(self, model1, model2):
        for my_p, p1, p2 in zip(self.model.parameters(), model1.parameters(), model2.parameters()):
            my_p.data.mul_(0).add_(0.5, p1.data).add_(0.5, p2.data)


class MeanSWAWeightsOptimizer(object):
    def __init__(self, model):
        self.model = model
        self.n_models = 0

    def getNModels(self):
        return self.n_models

    def setNModels(self, n_models):
        self.n_models = n_models

    def update(self, model1, model2):
        self.n_models += 1
        alpha_inv = 1. / float(self.n_models)
        alpha = 1. - alpha_inv
        alpha_inv *= 0.5
        for my_p, p1, p2 in zip(self.model.parameters(), model1.parameters(), model2.parameters()):
            my_p.data.mul_(alpha).add_(alpha_inv, p1.data).add_(alpha_inv, p2.data)