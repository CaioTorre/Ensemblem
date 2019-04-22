from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import KFold
import random
import numpy as np

class Ensemble:
    def __init__(self, n_folds=5):
        self.models = []
        for n in range(n_folds):
            self.models.append({})
        #self.models = [{}] * n_folds
        self.datasets = {
            'original': {'x':[], 'y':[], 'i':[]},
            'split': []
        }
        self.links = []
        for n in range(n_folds):
            self.datasets['split'].append({
                'train':    {'x':[], 'y':[], 'i':[]},
                'test':     {'x':[], 'y':[], 'i':[]}
            })
            self.links.append({'train':[], 'test':[]})
            
        self.encoder = None
        self.tokenizer = None
        self.n_folds = n_folds
        self.skfold = KFold(n_splits = n_folds, shuffle=True, random_state=1)
        #self.raw_avg = None
        #self.weight_sum = 0
        
    def add(self, name, model, weight=1, uses_one_hot=True, uses_categorical=False, uses_argmax=False):
        #if self.raw_avg == None:
        #    if weight == None:
        #        self.raw_avg = True
        #    else:
        #        self.raw_avg = False
        #else:
        #    if (self.raw_avg and weight != None):
        #        raise ValueError("Ensembled configured to not use weights but received weight value!")
        #    if (not self.raw_avg and weight == None):
        #        raise ValueError("Ensembled configured to use weights and current weight is None!")
        if name in self.models:
            raise ValueError("Ensemble already has a model named " + name)
        
        #self.weight_sum += weight
        for n in range(self.n_folds):
            self.models[n][name] = {'model':model, 'model_one_hot':uses_one_hot, 'model_categorical':uses_categorical, 'model_argmax':uses_argmax, 'last_predicts':[], 'weight':weight}
        
    def set_datasets(self, x, y, indexes):
        self.datasets['original']['x'] = x
        self.datasets['original']['y'] = y
        self.datasets['original']['i'] = indexes
        
    def split_data(self, shuffle_seed=None, purge_duplicates=True):
        if self.datasets['original']['y'] == []:
            raise RuntimeError("No dataset configured - use Ensemble.set_datasets(x, y)")
        if self.tokenizer == None:
            raise RuntimeError("Tokenizer is not configured - use Ensemble.set_tokenizer(new_tokenizer)")
        if self.encoder == None:
            raise RuntimeError("Encoder is not configured - use Ensemble.set_encoder(new_encoder)")
        
        if purge_duplicates:
            _x = self.tokenizer.texts_to_sequences(self.datasets['original']['x'])[:]
            _y = self.encoder.transform(self.datasets['original']['y']).tolist()[:]
            _i = self.datasets['original']['i'][:]
            print("Purging equal vectors...")
            purge_count = 0
            current_index = 0
            while (current_index < len(_x)):
                while (_x.count(_x[current_index]) > 1):
                    _x.pop(current_index)
                    _y.pop(current_index)
                    self.datasets['original']['x'].pop(current_index)
                    self.datasets['original']['y'].pop(current_index)
                    self.datasets['original']['i'].pop(current_index)
                    purge_count += 1
                current_index += 1
            print("Purged {} elements".format(purge_count))
        else:
            _x = self.tokenizer.texts_to_sequences(self.datasets['original']['x'])[:]
            _y = self.encoder.transform(self.datasets['original']['y']).tolist()[:]
            _i = self.datasets['original']['i'][:]
        
        #skfind = [index for index in self.skfold]
        
        n = 0
        for train_indexes, test_indexes in self.skfold.split(_x, _y, _i):
            self.datasets['split'][n]['train']['x'] = [_x[v] for v in train_indexes]
            self.datasets['split'][n]['train']['y'] = [_y[v] for v in train_indexes]
            self.datasets['split'][n]['test']['x'] = [_x[v] for v in test_indexes]
            self.datasets['split'][n]['test']['y'] = [_y[v] for v in test_indexes]
            self.links[n]['train'] = train_indexes[:]
            self.links[n]['test']  = test_indexes[:]
            n += 1
        
    def get_string_from_tokenizer(self, vec):
        return dict(map(vec, self.tokenizer.word_index.items()))
        
    def test_sets(self, ignore_collisions=True):
        print("Testing datasets for discrepancies...")
        for n in range(self.n_folds):
            collision_count = 0
            _baseline_x = self.tokenizer.texts_to_sequences(self.datasets['original']['x'])
            _baseline_y = self.encoder.transform(self.datasets['original']['y']).tolist()
            
            current_index = 0
            while (current_index < len(_baseline_x)):
                item = _baseline_x[current_index]
                #print(str(self.datasets['split'][n]['train']['x'] + self.datasets['split'][n]['test']['x']))
                item_index = (self.datasets['split'][n]['train']['x'] + self.datasets['split'][n]['test']['x']).index(item)
                if (int((self.datasets['split'][n]['train']['y'] + self.datasets['split'][n]['test']['y'])[item_index]) != _baseline_y[current_index]):
                    cnt = (self.datasets['split'][n]['train']['x'] + self.datasets['split'][n]['test']['x']).count(item)
                    if (cnt > 1):
                        print("More than one version of vector found in set - {} to be exact...".format(cnt))
                        collision_count += 1
                    if (not ignore_collisions) or (cnt == 1):
                        raise ValueError("Test failed on {}-th fold\nCurrent_index: {}\nExpected value: {}".format(n+1, current_index, _baseline_y[current_index]))
                current_index += 1
            print("{} fold checks out!".format(self.to_ordinal(n+1)))
            print("Finished fold with {} collisions!".format(collision_count))
        print("Test succeeded!")
            
    
    def set_encoder(self, new_encoder):
        old_encoder = self.encoder
        self.encoder = new_encoder
        return old_encoder
        
    def fit_encoder(self):
        if self.datasets['original']['y'] == []:
            raise RuntimeError("No dataset configured - use Ensemble.set_datasets(x, y)")
        if self.tokenizer == None:
            raise RuntimeError("Encoder is not configured - use Ensemble.set_encoder(new_encoder)")
        self.encoder.fit(self.datasets['original']['y'])
        
    def set_tokenizer(self, new_tokenizer):
        old_tokenizer = self.tokenizer
        self.tokenizer = new_tokenizer
        return old_tokenizer
        
    def fit_tokenizer(self):
        if self.datasets['original']['x'] == []:
            raise RuntimeError("No dataset configured - use Ensemble.set_datasets(x, y)")
        if self.tokenizer == None:
            raise RuntimeError("Tokenizer is not configured - use Ensemble.set_tokenizer(new_tokenizer)")
        self.tokenizer.fit_on_texts(self.datasets['original']['x'])
        
    def train(self, max_features, maxlen, profiler=None):
        if self.datasets['original']['x'] == []:
            raise RuntimeError("No dataset configured - use Ensemble.set_datasets(x, y)")
        if self.datasets['split'][0]['train']['x'] == []:
            raise RuntimeError("Dataset not split - use Ensemble.split_data(shuffle_seed=None)")
        if self.models == {}:
            raise RuntimeError("Ensemble is empty - add models via Ensemble.add(model, name, uses_one_hot=True, uses_categorical=False, uses_argmax=True)")
            
        if profiler is not None:
            profiler.add_milestone("Starting training")
            
        for n in range(self.n_folds):
            print("Training with the {} set".format(self.to_ordinal(n + 1)))
            _x_one_hot = []
            for vec in self.datasets['split'][n]['train']['x']:
                _x_one_hot.append([0.] * max_features)
                for _element in vec:
                    _x_one_hot[-1][_element] += 1.
            _x_one_hot = np.array(_x_one_hot)
            
            _x_orig = sequence.pad_sequences(np.array(self.datasets['split'][n]['train']['x']), maxlen=maxlen)
            
            _y_categorical = np_utils.to_categorical(self.datasets['split'][n]['train']['y'])
            
            for _m_key, _m_model in self.models[n].items():
                if profiler is not None:
                    profiler.add_milestone("Fold {} - Training model {}".format(n + 1, _m_key))
                else:
                    print("\tTraining model " + _m_key)
                    
                if _m_model['model_one_hot']:
                    _x = _x_one_hot
                else:
                    _x = _x_orig
                    
                if _m_model['model_categorical']:
                    _y = _y_categorical
                else:
                    _y = self.datasets['split'][n]['train']['y']
                    
                _m_model['model'].fit(_x, _y)
                
                if profiler is not None:
                    profiler.add_milestone("Fold {} - Trained model {}".format(n + 1, _m_key))
                
        print("Finished training")
        
    def test(self, max_features, maxlen, profiler=None):
        if profiler is not None:
            profiler.add_milestone("Starting testing")

        _predicts = [{}] * self.n_folds
        for n in range(self.n_folds):
            print("Testing with the {} set".format(self.to_ordinal(n + 1)))
            _x_one_hot = []
            for vec in self.datasets['split'][n]['test']['x']:
                _x_one_hot.append([0.] * max_features)
                for _element in vec:
                    _x_one_hot[-1][_element] += 1.
            _x_one_hot = np.array(_x_one_hot)
            
            _x_orig = sequence.pad_sequences(np.array(self.datasets['split'][n]['test']['x']), maxlen=maxlen)
            
            for _m_key, _m_model in self.models[n].items():
                if profiler is not None:
                    profiler.add_milestone("Fold {} - Testing model {}".format(n + 1, _m_key))
                else:
                    print("\tTesting model " + _m_key)
                    
                if _m_model['model_one_hot']:
                    _x = _x_one_hot
                else:
                    _x = _x_orig
                    
                _m_model['last_predicts'] = _m_model['model'].predict(_x)
                
                if _m_model['model_argmax']:
                    _m_model['last_predicts'] = np.array([np.argmax(y) for y in _m_model['last_predicts']])
                _predicts[n][_m_key] = _m_model['last_predicts']
                
                if profiler is not None:
                    profiler.add_milestone("Fold {} - Tested model {}".format(n + 1, _m_key))
                    
        print("Finished testing")
        return _predicts
        
    def evaluate(self, verbose=False, print_individual_results=True):
        _avgs = {'Ensemble':0.0}
        for n in range(self.n_folds):
            print("Evaluating {}-th fold".format(n + 1))
            _correct_guesses = {'Major': 0}
            for key in self.models[n]:
                _correct_guesses[key] = 0
                
            if verbose:
                header_msg = "\t     -  Correct "
                for key in self.models[n]:
                    header_msg += "- %8s " % (key)  
                header_msg += "- %8s" % ("Major.")
                print(header_msg + "- Phrase")
                
            total = len(self.datasets['split'][n]['test']['y'])
            for this_n in range(total - 1):
                _correct_predict = self.datasets['split'][n]['test']['y'][this_n]
                _correct_predict_strn = self.encoder.inverse_transform([_correct_predict])[0]
                msg = "\t%4d - %10s " % (this_n, _correct_predict_strn)
                votes = []
                
                for _m_key, _m_model in self.models[n].items():
                    model_vote = int(_m_model['last_predicts'][this_n])
                    for _ in range(_m_model['weight']):
                        votes.append(model_vote)
                    if model_vote == _correct_predict:
                        _correct_guesses[_m_key] += 1
                        msg += "- %10s* " % (self.encoder.inverse_transform([model_vote])[0])
                    else:
                        msg += "- %10s  " % (self.encoder.inverse_transform([model_vote])[0])
                        
                        
                major_vote = max(votes, key=votes.count)
                if major_vote == _correct_predict:
                    _correct_guesses['Major'] += 1
                    msg += "- %10s* " % (self.encoder.inverse_transform([major_vote])[0])
                else:
                    msg += "- %10s  " % (self.encoder.inverse_transform([major_vote])[0])
                
                if verbose:
                    print(msg + self.datasets['original']['x'][self.links[n]['test'][this_n]])
                    
            if print_individual_results:
                for name in self.models[n]:
                    print("\t" + name + " got {} out of {} ({}%)".format(_correct_guesses[name], total, 100. * _correct_guesses[name] / total))
                    try:
                        _avgs[name] += _correct_guesses[name] / total
                    except KeyError:
                        _avgs[name] = _correct_guesses[name] / total
                print("\tEnsemble got {} out of {} ({}%)".format(_correct_guesses['Major'], total, 100. * _correct_guesses['Major'] / total))
                _avgs['Ensemble'] += _correct_guesses['Major'] / total
        
        if print_individual_results:
            print("Across {} folds, averages are:".format(self.n_folds))
            for name in self.models[0]:
                print("\t" + name + ": %.3f%%" % (100. * _avgs[name] / self.n_folds))
            print("\tEnsemble: %.3f%%" % (100. * _avgs['Ensemble'] / self.n_folds))
        return total, _correct_guesses
        
    def to_ordinal(self, num):
        if num % 10 == 1:
            return '{}-st'.format(num)
        if num % 10 == 2:
            return '{}-nd'.format(num)
        if num % 10 == 3:
            return '{}-rd'.format(num)
        return '{}-th'.format(num)
        
