/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define PI 3.14159265358979323846

void ParticleFilter::init(const double x, const double y, const double theta, const double std[]) {

	// create normal distribution for initial values
	normal_distribution<double> normdist_x(x, std[0]);
	normal_distribution<double> normdist_y(y, std[1]);
	normal_distribution<double> normdist_theta(theta, std[2]);

	default_random_engine generator;

	// initialize particles with normal distributed samples
	for (int id = 0; id < num_particles; ++id) {
		
		Particle p;
		p.id = id;
		p.x = normdist_x(generator);
		p.y = normdist_y(generator);
		p.theta = normdist_theta(generator);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate) {

	default_random_engine generator;

	for (Particle &p : particles) {

		// predict the particles next position
		if(fabs(yaw_rate) < 0.01) {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else {
			p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}

		// add gaussian noise
		normal_distribution<double> normdist_x(0.0, std_pos[0]);
		normal_distribution<double> normdist_y(0.0, std_pos[1]);
		normal_distribution<double> normdist_theta(0.0, std_pos[2]);

		p.x += normdist_x(generator);
		p.y += normdist_y(generator);
		p.theta += normdist_theta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) const {

	// find the nearest landmark for every observation
	for (LandmarkObs &lm_o : observations) {
		
		auto it_p = predicted.begin();

		double min_dist = dist(lm_o.x, lm_o.y, it_p->x, it_p->y);
		lm_o.id = it_p->id;
		++it_p;

		for (; it_p != predicted.end(); ++it_p) {
			
			double curr_dist = dist(lm_o.x, lm_o.y, it_p->x, it_p->y);

			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				lm_o.id = it_p->id;
			}
		}
	}
}

std::vector<LandmarkObs> ParticleFilter::transformObservations(const std::vector<LandmarkObs> observations, const Particle p) const
{
	/*
	 * obs_M: Observation in vehicle's coordinate system
	 * obs_P: Observation in map's coordinate system
	 * R: Rotation matrix
	 * r: Translation vector (position of particle in map's coordinates)
	 * theta: Heading
	 *
	 * obs_M = R * obs_P + r
	 *
	 * R=  cos(theta) -sin(theta)
	 *		 sin(theta)  cos(theta)
	 */

	const double ct = cos(p.theta);
	const double st = sin(p.theta);

	std::vector<LandmarkObs> transformed;

	for (const LandmarkObs &it : observations) {
		
		LandmarkObs lm;
		
		lm.x = it.x * ct - it.y * st + p.x;
		lm.y = it.x * st + it.y * ct + p.y;

		transformed.push_back(lm);
	}

	return transformed;
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[], 
		const std::vector<LandmarkObs> observations, const Map map_landmarks) {

	// some constants for weight calculation
	const double f1 = 1.0 / (2.0 * PI * std_landmark[0] * std_landmark[1]);
	const double d1 = 2.0 * pow(std_landmark[0], 2);
	const double d2 = 2.0 * pow(std_landmark[1], 2);

	weights.clear();

	double weight_sum = 0.0;


	for (Particle &p : particles) {
		vector<LandmarkObs> predicted;

		// gather all map's landmarks within sensor range
		for (const Map::single_landmark_s &lm_m : map_landmarks.landmark_list) {
			if(dist((double)lm_m.x_f, (double)lm_m.y_f, p.x, p.y) <= sensor_range) {
				LandmarkObs lm;
				lm.id = lm_m.id_i;
				lm.x = (double)lm_m.x_f;
				lm.y = (double)lm_m.y_f;

				predicted.push_back(lm);
			}
		}

		// transform observations to the map's coordinate system
		vector<LandmarkObs> transformed = transformObservations(observations, p);
		
		dataAssociation(predicted, transformed);

		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		// reset weight
		p.weight = 1.0;

		// calculate weight for each observed landmark to its nearest map landmark
		for (LandmarkObs lm_t : transformed) {

			// find the map landmark with the corresponding id
			auto it = find_if(predicted.begin(), predicted.end(), [lm_t] (const LandmarkObs &lm) {
				return lm.id == lm_t.id;
			});

			// calculate weight
			if (it != predicted.end()) {
				double w = f1 * exp(-(pow(lm_t.x - it->x, 2)/d1 + pow(lm_t.y - it->y, 2)/d2));
				p.weight *= w;
			}

			associations.push_back(lm_t.id);
			sense_x.push_back(lm_t.x);
			sense_y.push_back(lm_t.y);
		}

		SetAssociations(p, associations, sense_x, sense_y);

		weight_sum += p.weight;
	}

	// normalize weights
	for (Particle &p : particles) {
		p.weight /= weight_sum;

		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {

	vector<Particle> resampled;

	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> discr_distr(weights.begin(), weights.end());

	// draw particles according to its weight
	for (int i = 0; i < particles.size(); ++i) {
		Particle p = particles[discr_distr(gen)];

		resampled.push_back(p);
	}

	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
