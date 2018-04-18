import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


class BayesianModel:
    def __init__(self, lst):
        self.num_edge = len(lst)
        self.lst = lst
        self.parent_BM = 0
        self.child_BM = 0

    def get_parent(self):
        '''
        given a certain node
        :return: the parent of this node in a DataFrame, the column is parent, index are the node
        '''
        parent = []
        child = []
        for i in range(self.num_edge):
            parent.append(self.lst[i][0])
            child.append(self.lst[i][1])
        data_pa = {'Parent': parent}
        self.parent_BM = DataFrame(data_pa, index = child)
        return self.parent_BM

    def get_child(self):
        '''
        given a certain node
        :return: the child of a node in a DataFrame, the column is the child, index is the parent
        '''
        parent = []
        child = []
        for i in range(self.num_edge):
            parent.append(self.lst[i][0])
            child.append(self.lst[i][1])
        data_ch = {'Child': child}
        self.child_BM = DataFrame(data_ch, index=parent)
        return self.child_BM

    def get_node(self):
        lst = []
        lst.extend(self.get_child().index)
        lst.extend(self.get_parent().index)
        new_lst = []
        for item in lst:
            if item not in new_lst:
                new_lst.append(item)
        return new_lst

    @staticmethod
    def add_cpds(*cpds):
        cpd_lst = []
        for cpd in cpds:
            cpd_lst.append(cpd.tabular())
        return cpd_lst

    @staticmethod
    def get_elimination_order(pool):
        '''
        return the elimination order of a set of given variables
        :param pool: the set of cpds based on each of the variable
        :return:
        '''
        def getRows(cpd):
            return len(cpd.index.values)
        output = sorted(pool, key=getRows)
        sequence = []
        for i in range(len(output)):
            pos_num = len(output[i].columns.values)-2
            eliminate_var = output[i].columns.values[pos_num]
            sequence.append(eliminate_var)
        return sequence


class TabularCPD:
    def __init__(self, variable, variable_card, values, evidence=None, evidence_card=None):
        self.variable = variable
        self.variable_card = variable_card
        self.values = values
        self.evidence = evidence
        self.evidence_card = evidence_card

    def tabular(self):
        prob_dict = {}
        prob_lst = []
        if self.evidence is None:
            prob_dict[self.variable] = range(self.variable_card)
            for i in range(self.variable_card):
                prob_lst.append(self.values[0][i])
            prob_dict['Prob'] = prob_lst
            table = DataFrame(prob_dict, index=range(self.variable_card))
        else:
            index_lst = self.get_index()
            length = len(index_lst[0])
            parent_lst = self.evidence
            parent_lst.append(self.variable)
            for i in range(len(parent_lst)):
                prob_dict[parent_lst[i]] = index_lst[i]
            prob_dict['Prob'] = self.values_reorder()
            parent_lst.append('Prob')
            table = DataFrame(prob_dict, index=range(length), columns=parent_lst)
        return table

    def values_reorder(self):
        value_lst = np.transpose(self.values)
        result_lst = []
        for i in range(len(value_lst)):
            result_lst.extend(value_lst[i])
        return result_lst

    def get_index(self):
        prod = 1
        card = self.evidence_card
        card.append(self.variable_card)
        for i in card:
            prod = prod*i
        index_outer_lst = []
        prod_for = 1
        for j in card:
            prod_for = prod_for * j
            index_inner_lst = []
            for k in range(j):
                for l in range(prod//prod_for):
                    index_inner_lst.append(k)
            index_outer_lst.append(index_inner_lst)
        result_lst = []
        for lst in index_outer_lst:
            inner_lst = []
            times = prod//len(lst)
            for i in range(times):
                inner_lst.extend(lst)
            result_lst.append(inner_lst)
        return result_lst


class VariableElimination:
    @staticmethod
    def query(sequence, query_var, pools, evidence=None):
        input_lst = sequence[:]
        input_lst.remove(query_var)
        for var in input_lst:
            table_join_input = []
            left = []
            for i in range(len(pools)):
                current_pool = pools[i]
                if var in list(current_pool.columns.values):
                    table_join_input.append(current_pool)
                else:
                    left.append(current_pool)
            table = VariableElimination.table_join(table_join_input, var, evidence)
            left.append(table)
            pools = left
        result = pools[0]
        total = result['Prob'].sum()
        result = result.Prob/total
        print(result.to_frame(name=query_var))

    @staticmethod
    def table_join(pool, var, evidence=None):
        '''

        :param pool: a list of dataframes that contains the variable to be eliminated next
        :param var:  the variable to be eliminated next
        :param evidence: the observed values, in the form of the dictionary
        :return: a dataframe taking into account the evidence with variable-to-be-eliminated eliminated
        '''
        table = pool[0]
        if evidence is not None:
            evi_var = list(evidence.keys())
            evi_val = list(evidence.values())
        else:
            evi_val = []
            evi_var = []
        for i in range(len(pool)-1):
            table = pd.merge(table, pool[i+1], on=var)
            table['Prob'] = table.Prob_x * table.Prob_y
            table.drop(columns=['Prob_x', 'Prob_y'], inplace=True)
        remain_lst = list(table.columns.values)
        remain_lst.remove('Prob')
        for i in range(len(evi_var)):
            if evi_var[i] in list(table.columns.values):
                table = table.loc[table[evi_var[i]] == evi_val[i]]
            else:
                continue
        remain_lst.remove(var)
        table = table.groupby(remain_lst)['Prob'].agg('sum').reset_index()
        return table


class Gibbs:
    def __init__(self, evidence=None):
        self.lst = ['X1', 'S1', 'X2', 'S2', 'X3', 'S3']
        if evidence is not None:
            self.evi_var = list(evidence.keys())
            self.evi_val = list(evidence.values())
            self.evidence = evidence
        else:
            self.evi_var = []

    @staticmethod
    def sample(threshold=None):
        if threshold is not None:
            hold = threshold
        else:
            hold = 0.5
        value = np.random.random_sample()
        if value >= hold:
            sample = 1
        else:
            sample = 0
        return sample

    def init_sampling(self):
        inner_lst = []
        for i in range(len(self.lst)):
            if self.lst[i] in self.evi_var:
                inner_lst.append(self.evidence[self.lst[i]])
            else:
                inner_lst.append(Gibbs.sample())
        return inner_lst

    @staticmethod
    def get_index(var, cpds):
        '''
        this function returns the position of cpd concerning var in cpds
        if x1 is the variable of interest, then the position of cpds that contains x1 is returned
        :param var: the var to be queried
        :param cpds: cpd pools
        :return: a list of index values based on which cpd of interests could be selected from cpd_pools
        '''
        column_names = []
        for cpd in cpds:
            name = list(cpd.columns.values)
            name.remove('Prob')
            column_names.append(name)
        index_values = []
        for i in range(len(column_names)):
            if var in column_names[i]:
                index_values.append(i)
        return index_values

    @staticmethod
    def trim_evidence(var, cpds, evidence_from_pre_sample):
        index = Gibbs.get_index(var, cpds)
        lst = []
        for i in index:
            lst.extend(list(cpds[i].columns.values))
        name_lst = []
        for name in lst:
            if name not in name_lst:
                name_lst.append(name)
        name_lst.remove('Prob')
        name_lst.remove(var)
        evidence_new = {}
        for key in name_lst:
            evidence_new[key] = evidence_from_pre_sample[key]
        return evidence_new

    @staticmethod
    def get_0_threshold(var, cpds, evidence_from_pre_sample):
        '''
        this function returns the possibility of variable being 0 given the evidence from the previous sample
        :param var: variable to be sampled
        :param cpds: cpd containing the variable in it
        :param evidence_from_pre_sample: the evidence from the previous sample
        :return: a float number which is the possibility of variable being 0
        '''
        index = Gibbs.get_index(var, cpds)
        cpd_concerned = []
        for i in index:
            cpd_concerned.append(cpds[i])
        table = cpd_concerned[0]
        for j in range(len(cpd_concerned)-1):
            table = pd.merge(table, cpd_concerned[j+1], on=var)
        evidence_new = Gibbs.trim_evidence(var, cpds, evidence_from_pre_sample)
        evi_var = list(evidence_new.keys())
        evi_val = list(evidence_new.values())
        # print(evidence_new)
        for k in range(len(evidence_new)):
            table = table.loc[table[evi_var[k]] == evi_val[k]]
        selected = table.loc[table[var] == 0]      #where the input var has the value of 0
        array = np.array(selected)[0]
        # print(array)
        prob_array = array[np.isin(array, [0, 1]) == False]
        numerator = prob_array.prod()
        # print(numerator)
        prob_df = VariableElimination.table_join(cpd_concerned, var, evidence=evidence_new)
        denominator = float(prob_df['Prob'])
        threshold = numerator/denominator
        return threshold

    def get_samples(self, time, cpds):
        samples = []
        samples.append(Gibbs.init_sampling(self))
        for i in range(1, time+1):
            inner_lst = []
            previous_sample = samples[i - 1][:]
            for j in range(len(self.lst)):
                variable_lst = self.lst[:]
                sample_var = variable_lst.pop(j)
                values_var_removed = []
                for k in range(6):
                    if k !=j:
                        values_var_removed.append(previous_sample[k])
                evidence_from_pre_sample = dict(zip(variable_lst, values_var_removed))
                if sample_var in self.evi_var:
                    sample_result = self.evidence[sample_var]
                else:
                    sample_result = Gibbs.sample(Gibbs.get_0_threshold(sample_var, cpds, evidence_from_pre_sample))
                previous_sample[j] = sample_result
                inner_lst.append(sample_result)
            samples.append(inner_lst)
        return samples

class Evaluation:
    def __init__(self, samples_input, evidence=None):
        self.sample = np.array(samples_input)
        self.lst = ['X1', 'S1', 'X2', 'S2', 'X3', 'S3']
        if evidence is not None:
            self.evidence = evidence
        else:
            self.evidence = []

    def get_prob(self):
        time_lst = []
        prob_lst = []
        step = 5
        for time in range(1, len(self.sample), step):
            selected_samples = self.sample[:time]
            match = selected_samples[selected_samples[:, 4] == 0]
            prob = match.shape[0]/time
            prob_lst.append(prob)
            time_lst.append(time)
        plt.plot(time_lst, prob_lst, 'b-')
        if len(self.evidence) ==0 :
            plt.axhline(y=0.36, color='r', linestyle='--')
        elif len(self.evidence) == 1:
            plt.axhline(y=0.46, color='r', linestyle='--')
        else:
            plt.axhline(y=0.33, color='r', linestyle='--')
        plt.show()

if __name__ == '__main__':
    model = BayesianModel([('X1', 'S1'), ('X1', 'X2'), ('X2', 'S2'), ('X2', 'X3'), ('X3', 'S3')])
    cpd_x1 = TabularCPD(variable='X1', variable_card=2, values=[[0.1, 0.9]])
    cpd_s1 = TabularCPD(variable='S1', variable_card=2, values=[[0.9, 0.5],
                                                                [0.1, 0.5]],
                        evidence=['X1'], evidence_card=[2])
    cpd_x2 = TabularCPD(variable='X2', variable_card=2, values=[[0.9, 0.2],
                                                                [0.1, 0.8]],
                        evidence=['X1'], evidence_card=[2])
    cpd_s2 = TabularCPD(variable='S2', variable_card=2, values=[[0.9, 0.5],
                                                                [0.1, 0.5]],
                        evidence=['X2'], evidence_card=[2])
    cpd_x3 = TabularCPD(variable='X3', variable_card=2, values=[[0.9, 0.2],
                                                                [0.1, 0.8]],
                        evidence=['X2'], evidence_card=[2])
    cpd_s3 = TabularCPD(variable='S3', variable_card=2, values=[[0.9, 0.5],
                                                                [0.1, 0.5]],
                        evidence=['X3'], evidence_card=[2])
    cpd_pools = model.add_cpds(cpd_x1, cpd_s1, cpd_x2, cpd_s2, cpd_x3, cpd_s3)
    elimination_order = model.get_elimination_order(cpd_pools)
    infer = VariableElimination
    gibbs1 = Gibbs(evidence=None)
    samples = gibbs1.get_samples(2000, cpd_pools)
    eva = Evaluation(samples, evidence=None)
    eva.get_prob()
    gibbs2 = Gibbs(evidence={'S2': 0})
    samples2 =gibbs2.get_samples(2000, cpd_pools)
    eva2 = Evaluation(samples2, evidence={'S2': 0})
    eva2.get_prob()
    gibb3 = Gibbs(evidence={'S1':0, 'S2':1, 'S3':0})
    sample3 = gibb3.get_samples(2000, cpd_pools)
    eva3 = Evaluation(sample3,evidence={'S1':0, 'S2':1, 'S3':0})
    eva3.get_prob()



