# -*- coding: utf-8 -*-
# A simple re-implement of https://arxiv.org/abs/1802.07068
# Ref: [1]Rapisarda A. Talent vs Luck: the role of randomness in success and failure[J]. 2018.

import numpy as np
import draw_result
import matplotlib.pyplot as plt
import time
import utils


class Individual:
    # Implement Individual class.
    def __init__(self, name, talent, init_capital, location):
        self.name = name
        self.talent = float(talent)
        self.capital = float(init_capital)
        self.location = location
        self.incident_record = []
        self.full_incident_record = []
        self.lucky_incident_num = 0
        self.unlucky_incident_num = 0

    def encounter_incident(self, incident_time, incident_location, is_lucky, talent_boundary):
        """
        Update the capital according to possible incident. Encounter means distance between individual
        and incident is <= 1. incident_record is a list. Each element is a list:[incident_time, flag, capital].
        flag is a signal of incident type. 1 indicates one gain capital, 0 indicates no changes,
        -1 indicates one lose capital.
        :param incident_time: Incident happen time
        :param incident_location: Current incident location
        :param is_lucky: Boolean. If lucky, the value is True, else is False
        :param talent_boundary: Judge whether an individual can benefit from an incident
        :return:
        """
        if np.linalg.norm(self.location - incident_location) <= 1:
            if is_lucky:
                # talent_boundary is the lower boundary that an individual can benefit from a lucky incident
                if self.talent > talent_boundary:
                    self.capital = self.capital * 2
                    self.incident_record.append([incident_time, 1, self.capital])
                    self.lucky_incident_num = self.lucky_incident_num + 1
            else:
                self.capital = self.capital / 2
                self.incident_record.append([incident_time, -1, self.capital])
                self.unlucky_incident_num = self.unlucky_incident_num + 1

    def get_full_incident_record(self, incident_num, init_capital):
        """
        Generate the full incident and capital record of an individual.
        :param incident_num: total incident num, should be equal to sim_time / time_interval
        :param init_capital:initial capital
        :return: full_incident_record. First col is incident number. Second column is incident type. 0 means
        no incident. 1 means lucky incident. -1 means unlucky incident. Third column is capital record.
        """
        full_incident_record = np.zeros((incident_num, 3), dtype=float)
        full_incident_record[:, 2] = init_capital * np.ones(full_incident_record.shape[0], dtype=float)
        full_incident_record[:, 0] = np.array([(i + 1) for i in range(1, incident_num + 1)])
        # fill the full record with list record.
        for record in self.incident_record:
            full_incident_record[record[0] - 1, :] = record
        for i in range(1, incident_num):
            # no incident happens to an individual. his capital remains same.
            if np.abs(full_incident_record[i, 2] - 10) < 1e-10:
                full_incident_record[i, 2] = full_incident_record[i - 1, 2]
        self.full_incident_record = full_incident_record
        return full_incident_record


class Incident:
    # incident time starts from 1
    def __init__(self, name, init_location, islucky):
        self.name = name
        self.time = 0
        self.location = init_location
        self.islucky = islucky

    def move(self, length, direction, max_x_boundary, max_y_boundary):
        # Generate next location of this incident. If the location is out of range, use that axis as a symmetry axis
        # and reflect it into the range. Minimum boundary is set to 0 by default.
        if self.location[0] + length * np.cos(direction * np.pi / 180) > max_x_boundary:
            self.location[0] = 2 * max_x_boundary - (self.location[0] + length * np.cos(direction * np.pi / 180))
        elif self.location[1] + length * np.sin(direction * np.pi / 180) > max_y_boundary:
            self.location[1] = 2 * max_y_boundary - (self.location[1] + length * np.sin(direction * np.pi / 180))
        else:
            self.location = np.abs([self.location[0] + length * np.cos(direction * np.pi / 180),
                                    self.location[1] + length * np.sin(direction * np.pi / 180)])


def main():
    # environment parameter setting
    sim_time = 40
    time_interval = 0.5
    max_boundary_x = 201
    max_boundary_y = 201

    # individual parameter setting
    individual_num = 1000
    talent_distribution = 'truncated_normal'
    individual_location_distribution = 'uniform'
    talent_mean = 0.6
    talent_std = 0.1
    initial_capital = 10

    # incident parameter setting
    incident_num = 500
    incident_location_distribution = 'uniform'
    lucky_percentage = 0.5
    move_length = 2

    # save result setting
    savepath = './result/'
    image_format = '.png'
    draw_result.save_init(savepath)

    def singlerun(save_result=False, show_result=True):
        """
        run a single simulation process
        :param save_result: whether to save simulation result
        :param show_result: whether to show simulation result
        :return: the richest people's talent
        """

        # initial the talent, location distribution and incident move direction set. Direction unit is degree.
        talent_set = utils.generate_distribution(size=individual_num,
                                                 distribution_type=talent_distribution,
                                                 data_type='float',
                                                 min_val=0,
                                                 max_val=1,
                                                 mean=talent_mean,
                                                 std=talent_std)
        individual_location_set = utils.generate_location(size=individual_num,
                                                          distribution_type=individual_location_distribution,
                                                          data_type='float',
                                                          min_val=0,
                                                          max_val_x=max_boundary_x,
                                                          max_val_y=max_boundary_y)
        incident_location_set = utils.generate_location(size=incident_num,
                                                        distribution_type=incident_location_distribution,
                                                        data_type='float',
                                                        min_val=0,
                                                        max_val_x=max_boundary_x,
                                                        max_val_y=max_boundary_y)
        direction_set = utils.generate_distribution(size=(int(sim_time / time_interval), incident_num),
                                                    distribution_type='uniform',
                                                    data_type='int',
                                                    min_val=0,
                                                    max_val=360)

        # initial the individual and incident instances list
        individual_set = [Individual(i, talent_set[i], initial_capital, individual_location_set[i])
                          for i in range(individual_num)]
        incident_set = ([Incident(j, incident_location_set[j], True)
                        for j in range(int(incident_num * lucky_percentage))] +
                        [Incident(k, incident_location_set[k], False)
                        for k in range(int(incident_num * lucky_percentage), incident_num)])

        # Start the simulation
        start = time.time()
        for epoch in range(int(sim_time / time_interval)):
            # all incidents take a random direction move
            for inc_index in range(incident_num):
                incident_set[inc_index].move(length=move_length,
                                             direction=direction_set[epoch, inc_index],
                                             max_x_boundary=max_boundary_x,
                                             max_y_boundary=max_boundary_y)
                # check each individual whether he/she encounters an incident
                for ind_index in range(individual_num):
                    individual_set[ind_index].encounter_incident(incident_time=epoch + 1,
                                                                 incident_location=incident_set[inc_index].location,
                                                                 is_lucky=incident_set[inc_index].islucky,
                                                                 talent_boundary=np.random.rand())
            print('No.%d incidents happen.' % (epoch + 1))
        end = time.time()
        runtime = end - start
        print('Simulation finishes. Total time:%dh%dm%ds' % (runtime // 3600, runtime % 3600 // 60,
                                                             runtime % 3600 % 60))

        # save result & draw figure
        final_capital_set = np.array([individual_set[i].capital for i in range(individual_num)])
        lucky_incident_set = np.array([individual_set[i].lucky_incident_num for i in range(individual_num)])
        unlucky_incident_set = np.array([individual_set[i].unlucky_incident_num for i in range(individual_num)])
        full_incident_set = np.array([individual_set[i].get_full_incident_record(int(sim_time / time_interval),
                                                                                 initial_capital)
                                      for i in range(individual_num)])
        if save_result:
            np.save(savepath + 'talent_set.npy', talent_set)
            np.save(savepath + 'individual_location_set.npy', individual_location_set)
            np.save(savepath + 'incident_location_set.npy', incident_location_set)
            np.save(savepath + 'final_capital_set.npy', final_capital_set)
            np.save(savepath + 'lucky_incident_set.npy', lucky_incident_set)
            np.save(savepath + 'unlucky_incident_set.npy', unlucky_incident_set)
        if show_result:
            draw_result.static_initial_location(individual_location_set,
                                                incident_location_set,
                                                max_boundary_x,
                                                max_boundary_y,
                                                lucky_percentage,
                                                savepath,
                                                image_format)
            draw_result.static_talent(talent_set, savepath, image_format)
            draw_result.static_capital_individual_num(final_capital_set, savepath, image_format)
            draw_result.static_capital_talent(final_capital_set, talent_set, savepath, image_format)
            draw_result.static_capital_incident(final_capital_set, lucky_incident_set, unlucky_incident_set, savepath,
                                                image_format)
            draw_result.select_richest_poorest(talent_set, final_capital_set, full_incident_set, savepath, image_format)
            plt.show()
        return individual_set[int(np.argmax(final_capital_set))].talent

    def multirun(save_result=False, show_result=True, epochs=500):
        richest_individual_talent_set = []
        for _ in range(epochs):
            richest_individual_talent_set.append(singlerun(save_result, show_result))
            print('%d epoch finish.' % (_ + 1))
        draw_result.static_multiple_richest_talent(richest_individual_talent_set, savepath, image_format)

    multirun(save_result=False, show_result=False, epochs=500)


if __name__ == '__main__':
    main()
