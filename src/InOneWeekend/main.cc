//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "random.h"
#include "sphere.h"

#include <float.h>
#include <iostream>

#include <fstream>
#include <chrono>
#include <string>

#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>

vec3 color(const ray& r, hittable* world, int depth) {
	hit_record rec;
	if (world->hit(r, 0.001, FLT_MAX, rec)) {
		ray scattered;
		vec3 attenuation;
		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
			return attenuation * color(scattered, world, depth + 1);
		}
		else {
			return vec3(0, 0, 0);
		}
	}
	else {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}


hittable* random_scene() {
	int n = 500;
	hittable** list = new hittable * [n + 1];
	list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = random_double();
			vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {  // diffuse
					list[i++] = new sphere(
						center, 0.2,
						new lambertian(vec3(random_double() * random_double(),
							random_double() * random_double(),
							random_double() * random_double()))
					);
				}
				else if (choose_mat < 0.95) { // metal
					list[i++] = new sphere(
						center, 0.2,
						new metal(vec3(0.5 * (1 + random_double()),
							0.5 * (1 + random_double()),
							0.5 * (1 + random_double())),
							0.5 * random_double())
					);
				}
				else {  // glass
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}

	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

	return new hittable_list(list, i);
}

std::mutex writeM;

struct BlockJob
{
	int rowStart = 0;
	int rowEnd = 0;
	int colSize;
	int spp;
	std::vector<int> indices;
	std::vector<vec3> colors;
};

void CalculateColor(BlockJob job, std::vector<BlockJob>& imageBlocks, int ny, camera cam, hittable* world,
	std::mutex& mutex, std::condition_variable& cv)
{
	for (int j = job.rowStart; j < job.rowEnd; ++j) {
		for (int i = 0; i < job.colSize; ++i) {
			vec3 col(0, 0, 0);
			for (int s = 0; s < job.spp; ++s) {
				float u = float(i + random_double()) / float(job.colSize);
				float v = float(j + random_double()) / float(ny);
				ray r = cam.get_ray(u, v);
				col += color(r, world, 0);
			}
			col /= float(job.spp);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			const unsigned int index = j * job.colSize + i;
			job.indices.push_back(index);
			job.colors.push_back(col);
		}
	}

	{
		std::lock_guard<std::mutex> lock(mutex);
		imageBlocks.push_back(job);
	}
}

void ThreadJobLoop(
	int ny, camera cam, hittable* world,
	std::queue<BlockJob>& jobQ, 
	std::vector<BlockJob>& finishedJobs, 
	std::mutex& mutex,
	std::condition_variable& cv
	)
{
	std::atomic<bool> hasWork{ true };
	while (hasWork)
	{
		BlockJob job;
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (!jobQ.empty())
			{
				job = jobQ.front();
				jobQ.pop();
			}
		}
		// quick/dirty way to find if a job is valid
		if (job.rowStart < job.rowEnd)
		{
			CalculateColor(job, finishedJobs, ny, cam, world, mutex, cv);
		}
		else
		{
			hasWork = false;
		}
	}

	// no more jobs.
	{
		std::lock_guard<std::mutex> lock(mutex);
		cv.notify_one();
	}
}

int main() {
	int nx = 1200;
	int ny = 800;
	int ns = 10;
	int pixelCount = nx * ny;
	// std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	hittable* world = random_scene();

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0f;
	float aperture = 0.1f;

	vec3* image = new vec3[pixelCount];
	memset(&image[0], 0, pixelCount * sizeof(vec3));

	camera cam(lookfrom, lookat, vec3(0, -1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);

	auto fulltime = std::chrono::high_resolution_clock::now();

	const int nThreads = std::thread::hardware_concurrency();
	int nRowsPerJob = 200;
	int nJobs = ny / nRowsPerJob; // 1 row per job
	int leftOver = ny % nThreads;

	std::mutex mutex;
	std::condition_variable cvResults;
	std::vector<BlockJob> imageBlocks;
	std::queue<BlockJob> jobQueue;
	std::vector<std::thread> threads;

	for (int i = 0; i < nJobs; ++i)
	{
		BlockJob job;
		job.rowStart = i * nRowsPerJob;
		job.rowEnd = job.rowStart + nRowsPerJob;
		if (i == nThreads - 1)
		{
			job.rowEnd = job.rowStart + nRowsPerJob + leftOver;
		}
		job.colSize = nx;
		job.spp = ns;

		jobQueue.push(job);
	}

	// last processing thread, is the main thread.
	for (int i = 0; i < nThreads - 1; ++i)
	{
		std::thread t([&]() {
			ThreadJobLoop(ny, cam, world, jobQueue, imageBlocks, mutex, cvResults);
		});
		threads.push_back(std::move(t));
	}

	// main thread loop
	ThreadJobLoop(ny, cam, world, jobQueue, imageBlocks, mutex, cvResults);

	// launched jobs. need to build image.
	// wait for number of jobs = pixel count
	{
		std::unique_lock<std::mutex> lock(mutex);
		cvResults.wait(lock, [&imageBlocks, &nJobs] {
			return imageBlocks.size() == nJobs;
		});
	}

	for (std::thread& t : threads)
	{
		t.join();
	}

	for (BlockJob job : imageBlocks)
	{
		int index = job.rowStart;
		int colorIndex = 0;
		for (vec3& col : job.colors)
		{
			int colIndex = job.indices[colorIndex];
			image[colIndex] = col;
			++colorIndex;
		}
	}

	auto timeSpan = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - fulltime);
	int frameTimeMs = static_cast<int>(timeSpan.count());
	std::cout << " - time " << frameTimeMs << " ms \n";

	std::string filename = 
		"block-jobq-x" + std::to_string(nx) 
		+ "-y" + std::to_string(ny) 
		+ "-s" + std::to_string(ns) 
		+ "-" + std::to_string(frameTimeMs) + "sec.ppm";

	std::ofstream fileHandler;
	fileHandler.open(filename, std::ios::out | std::ios::binary);
	if (!fileHandler.is_open())
	{
		return false;
	}

	fileHandler << "P3\n" << nx << " " << ny << "\n255\n";

	for (unsigned int i = 0; i < nx * ny; ++i)
	{
		// BGR to RGB
		// 2 = r;
		// 1 = g;
		// 0 = b;
		fileHandler
			<< static_cast<int>(255.99f * image[i].e[2]) << " "
			<< static_cast<int>(255.99f * image[i].e[1]) << " "
			<< static_cast<int>(255.99f * image[i].e[0]) << "\n";
	}

	std::cout << "File Saved" << std::endl;
	fileHandler.close();

	delete[] image;
	return 0;
}
