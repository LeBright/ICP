#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>


using namespace std;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

typedef struct {
	int i;	
	int j;
	double e_x1;
	double e_y1;
	double e_z1;
	double e_x2;
	double e_y2;
	double e_z2;
}point_e;

__global__ void Init_new_Q(double** P, double** Q, double** newQ, int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		int min_idx = 0;
		double min_val = (P[threadID][0] - Q[min_idx][0]) * (P[threadID][0] - Q[min_idx][0])
			+ (P[threadID][1] - Q[min_idx][1]) * (P[threadID][1] - Q[min_idx][1])
			+ (P[threadID][2] - Q[min_idx][2]) * (P[threadID][2] - Q[min_idx][2]);
		for (int j = 0; j < n; j++)
		{
			double temp_val = (P[threadID][0] - Q[j][0]) * (P[threadID][0] - Q[j][0])
				+ (P[threadID][1] - Q[j][1]) * (P[threadID][1] - Q[j][1])
				+ (P[threadID][2] - Q[j][2]) * (P[threadID][2] - Q[j][2]);
			if (temp_val < min_val)
			{
				min_idx = j;
				min_val = temp_val;
			}
		}
		newQ[threadID][0] = Q[min_idx][0];
		newQ[threadID][1] = Q[min_idx][1];
		newQ[threadID][2] = Q[min_idx][2];
	}
}
__global__ void calculate_center(double** P, double* P_x_center_device, double* P_y_center_device, double* P_z_center_device, int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		__shared__ double warp_x[32];
		__shared__ double warp_y[32];
		__shared__ double warp_z[32];
		double temp_x = 0;
		double temp_y = 0;
		double temp_z = 0;
		double x = P[threadID][0] / n;
		double y = P[threadID][1] / n;
		double z = P[threadID][2] / n;
		x += __shfl_xor_sync(0xffffffffUL, x, 16);
		x += __shfl_xor_sync(0xffffffffUL, x, 8);
		x += __shfl_xor_sync(0xffffffffUL, x, 4);
		x += __shfl_xor_sync(0xffffffffUL, x, 2);
		x += __shfl_xor_sync(0xffffffffUL, x, 1);
		y += __shfl_xor_sync(0xffffffffUL, y, 16);
		y += __shfl_xor_sync(0xffffffffUL, y, 8);
		y += __shfl_xor_sync(0xffffffffUL, y, 4);
		y += __shfl_xor_sync(0xffffffffUL, y, 2);
		y += __shfl_xor_sync(0xffffffffUL, y, 1);
		z += __shfl_xor_sync(0xffffffffUL, z, 16);
		z += __shfl_xor_sync(0xffffffffUL, z, 8);
		z += __shfl_xor_sync(0xffffffffUL, z, 4);
		z += __shfl_xor_sync(0xffffffffUL, z, 2);
		z += __shfl_xor_sync(0xffffffffUL, z, 1);



		if (threadIdx.x < 32 && threadIdx.x >= (blockDim.x >> 5))
		{
			warp_x[threadIdx.x] = 0;
			warp_y[threadIdx.x] = 0;
			warp_z[threadIdx.x] = 0;
		}

		if (threadIdx.x % 32 == 0)
		{
			warp_x[threadIdx.x / 32u] = x;
			warp_y[threadIdx.x / 32u] = y;
			warp_z[threadIdx.x / 32u] = z;

		}
		__syncthreads();

		if (threadIdx.x < 32)
		{
			temp_x = warp_x[threadIdx.x];
			temp_y = warp_y[threadIdx.x];
			temp_z = warp_z[threadIdx.x];
			temp_x += __shfl_xor_sync(0xffffffffUL, temp_x, 16);
			temp_x += __shfl_xor_sync(0xffffffffUL, temp_x, 8);
			temp_x += __shfl_xor_sync(0xffffffffUL, temp_x, 4);
			temp_x += __shfl_xor_sync(0xffffffffUL, temp_x, 2);
			temp_x += __shfl_xor_sync(0xffffffffUL, temp_x, 1);
			temp_y += __shfl_xor_sync(0xffffffffUL, temp_y, 16);
			temp_y += __shfl_xor_sync(0xffffffffUL, temp_y, 8);
			temp_y += __shfl_xor_sync(0xffffffffUL, temp_y, 4);
			temp_y += __shfl_xor_sync(0xffffffffUL, temp_y, 2);
			temp_y += __shfl_xor_sync(0xffffffffUL, temp_y, 1);
			temp_z += __shfl_xor_sync(0xffffffffUL, temp_z, 16);
			temp_z += __shfl_xor_sync(0xffffffffUL, temp_z, 8);
			temp_z += __shfl_xor_sync(0xffffffffUL, temp_z, 4);
			temp_z += __shfl_xor_sync(0xffffffffUL, temp_z, 2);
			temp_z += __shfl_xor_sync(0xffffffffUL, temp_z, 1);
		}

		if (threadIdx.x == 0)
		{
			P_x_center_device[blockIdx.x] = temp_x;
			P_y_center_device[blockIdx.x] = temp_y;
			P_z_center_device[blockIdx.x] = temp_z;
		}

	}
}
__global__ void mat_minus_vec(double** P, double* p, int n, bool trans = false)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		if (!trans)
		{
			P[threadID][0] -= p[0];
			P[threadID][1] -= p[1];
			P[threadID][2] -= p[2];
		}
		else
		{
			P[0][threadID] -= p[0];
			P[1][threadID] -= p[1];
			P[2][threadID] -= p[2];
		}
	}
}
__global__ void mat_add_vec(double** P, double* p, int n, bool trans = false)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		if (!trans)
		{
			P[threadID][0] += p[0];
			P[threadID][1] += p[1];
			P[threadID][2] += p[2];
		}
		else
		{
			P[0][threadID] += p[0];
			P[1][threadID] += p[1];
			P[2][threadID] += p[2];
		}
	}
}
__global__ void mat_mul(double** A, double** B, double** C, int num_points, int n, bool left_trans = false, bool right_trans = false)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		int group = threadID / num_points;
		int k = threadID % num_points;
		int row = group / 3;
		int col = group % 3;
		if (!left_trans && !right_trans)
		{
			atomicAdd(&C[row][col], A[row][k] * B[k][col]);
		}
		else if (!left_trans && right_trans)
		{
			atomicAdd(&C[row][col], A[row][k] * B[col][k]);
		}
		else if (left_trans && !right_trans)
		{
			atomicAdd(&C[row][col], A[k][row] * B[k][col]);
		}
		else
		{
			atomicAdd(&C[row][col], A[k][row] * B[col][k]);
		}
		
	}
}
__global__ void calculate_distance(double** P, double** Q, int n, double* distance_block)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		__shared__ double warp_result[32];
		double distance = ((P[threadID][0] - Q[threadID][0]) * (P[threadID][0] - Q[threadID][0])
			+ (P[threadID][1] - Q[threadID][1]) * (P[threadID][1] - Q[threadID][1])
			+ (P[threadID][2] - Q[threadID][2]) * (P[threadID][2] - Q[threadID][2])) / n;
		double temp = 0;
		distance += __shfl_xor_sync(0xffffffffUL, distance, 16);
		distance += __shfl_xor_sync(0xffffffffUL, distance, 8);
		distance += __shfl_xor_sync(0xffffffffUL, distance, 4);
		distance += __shfl_xor_sync(0xffffffffUL, distance, 2);
		distance += __shfl_xor_sync(0xffffffffUL, distance, 1);

		if (threadIdx.x < 32 && threadIdx.x >= (blockDim.x >> 5))
		{
			warp_result[threadIdx.x] = 0;
		}

		if (threadIdx.x % 32 == 0)
		{
			warp_result[threadIdx.x / 32u] = distance;
		}
		__syncthreads();

		if (threadIdx.x < 32)
		{
			temp = warp_result[threadIdx.x];
			temp += __shfl_xor_sync(0xffffffffUL, temp, 16);
			temp += __shfl_xor_sync(0xffffffffUL, temp, 8);
			temp += __shfl_xor_sync(0xffffffffUL, temp, 4);
			temp += __shfl_xor_sync(0xffffffffUL, temp, 2);
			temp += __shfl_xor_sync(0xffffffffUL, temp, 1);
		}

		if (threadIdx.x == 0)
			distance_block[blockIdx.x] = temp;
	}
}
__global__ void change(double** R, double** P, double* t, int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID < n)
	{
		double x = P[threadID][0];
		double y = P[threadID][1];
		double z = P[threadID][2];
		P[threadID][0] = x * R[0][0] + y * R[0][1] + z * R[0][2] + t[0];
		P[threadID][1] = x * R[1][0] + y * R[1][1] + z * R[1][2] + t[1];
		P[threadID][2] = x * R[2][0] + y * R[2][1] + z * R[2][2] + t[2];
	}
}
__global__ void find_points_R1(double** Q,int**set1,int**set2, double R1_first, double R1_second,unsigned int* set1_idx,unsigned int* set2_idx,int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n*n)
	{
		int i = threadID / n;
		int j = threadID % n;
		double x = Q[i][0] - Q[j][0];
		double y = Q[i][1] - Q[j][1];
		double z = Q[i][2] - Q[j][2];
		double R = sqrt(x * x + y * y + z * z);

		if (fabs(R - R1_first) < 0.0001)
		{
			unsigned int old=atomicAdd(set1_idx,1);
			if (old < n-1)
			{
				set1[old + 1][0] = i;
				set1[old + 1][1] = j;
			}
		}
		else if (fabs(R - R1_second) < 0.0001)
		{
			unsigned int old = atomicAdd(set2_idx,1);
			if (old < n-1)
			{
				set2[old + 1][0] = i;
				set2[old + 1][1] = j;
			}
		}
	}
}
__global__ void find_e(double** Q, int** set, double R2, point_e* e,int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n)
	{
		e[threadID].i = set[threadID][0];
		e[threadID].j = set[threadID][1];
		double A_x = Q[set[threadID][0]][0];
		double A_y = Q[set[threadID][0]][1];
		double A_z = Q[set[threadID][0]][2];
		double B_x = Q[set[threadID][1]][0];
		double B_y = Q[set[threadID][1]][1];
		double B_z = Q[set[threadID][1]][2];
		double lamda = R2 / (1 - R2);
		e[threadID].e_x1 = (A_x + lamda * B_x) / (1 + lamda);
		e[threadID].e_y1 = (A_y + lamda * B_y) / (1 + lamda);
		e[threadID].e_z1 = (A_z + lamda * B_z) / (1 + lamda);
		e[threadID].e_x2 = (B_x + lamda * A_x) / (1 + lamda);
		e[threadID].e_y2 = (B_y + lamda * A_y) / (1 + lamda);
		e[threadID].e_z2 = (B_z + lamda * A_z) / (1 + lamda);
	}
}
__global__ void final_four_points_set(point_e* first, point_e* second, int** result,double** e_point, int* count,int n)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < n*n)
	{
		int first_idx = threadID / n;
		int second_idx = threadID % n;
		if (!(first[first_idx].i == 0 && first[first_idx].j == 0)&& !(second[second_idx].i == 0 && second[second_idx].j == 0))
		{
			double distance1 = (first[first_idx].e_x1 - second[second_idx].e_x1) * (first[first_idx].e_x1 - second[second_idx].e_x1)
				+ (first[first_idx].e_y1 - second[second_idx].e_y1) * (first[first_idx].e_y1 - second[second_idx].e_y1)
				+ (first[first_idx].e_z1 - second[second_idx].e_z1) * (first[first_idx].e_z1 - second[second_idx].e_z1);
			double distance2 = (first[first_idx].e_x1 - second[second_idx].e_x2) * (first[first_idx].e_x1 - second[second_idx].e_x2)
				+ (first[first_idx].e_y1 - second[second_idx].e_y2) * (first[first_idx].e_y1 - second[second_idx].e_y2)
				+ (first[first_idx].e_z1 - second[second_idx].e_z2) * (first[first_idx].e_z1 - second[second_idx].e_z2);
			double distance3 = (first[first_idx].e_x2 - second[second_idx].e_x1) * (first[first_idx].e_x2 - second[second_idx].e_x1)
				+ (first[first_idx].e_y2 - second[second_idx].e_y1) * (first[first_idx].e_y2 - second[second_idx].e_y1)
				+ (first[first_idx].e_z2 - second[second_idx].e_z1) * (first[first_idx].e_z2 - second[second_idx].e_z1);
			double distance4 = (first[first_idx].e_x2 - second[second_idx].e_x2) * (first[first_idx].e_x2 - second[second_idx].e_x2)
				+ (first[first_idx].e_y2 - second[second_idx].e_y2) * (first[first_idx].e_y2 - second[second_idx].e_y2)
				+ (first[first_idx].e_z2 - second[second_idx].e_z2) * (first[first_idx].e_z2 - second[second_idx].e_z2);

			if (distance1 < 0.0000001 || distance2 < 0.0000001 || distance3 < 0.0000001 || distance4 < 0.0000001)
			{
				int old = atomicAdd(count, 1);
				if (old < n - 1)
				{
					result[old+1][0] = first[first_idx].i;
					result[old+1][1] = first[first_idx].j;
					result[old+1][2] = second[second_idx].i;
					result[old+1][3] = second[second_idx].j;
					if (distance1 < 0.0000001)
					{
						e_point[old + 1][0] = first[first_idx].e_x1;
						e_point[old + 1][1] = first[first_idx].e_y1;
						e_point[old + 1][2] = first[first_idx].e_z1;
					}
					else if (distance2 < 0.0000001)
					{
						e_point[old + 1][0] = first[first_idx].e_x1;
						e_point[old + 1][1] = first[first_idx].e_y1;
						e_point[old + 1][2] = first[first_idx].e_z1;
					}
					else if (distance3 < 0.0000001)
					{
						e_point[old + 1][0] = first[first_idx].e_x2;
						e_point[old + 1][1] = first[first_idx].e_y2;
						e_point[old + 1][2] = first[first_idx].e_z2;
					}
					else
					{
						e_point[old + 1][0] = first[first_idx].e_x2;
						e_point[old + 1][1] = first[first_idx].e_y2;
						e_point[old + 1][2] = first[first_idx].e_z2;
					}
				}
			}
		}
	}
}


double calculate_distance(const vector<double>& P, const vector<double>& Q)
{
	return sqrt((P[0] - Q[0]) * (P[0] - Q[0]) + (P[1] - Q[1]) * (P[1] - Q[1]) + (P[2] - Q[2]) * (P[2] - Q[2]));
}
double calculate_set_distance(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	int n = A.size();
	double ans = 0;
	for (int i = 0; i < n; i++)
	{
		ans += calculate_distance(A[i], B[i]);
	}
	ans /= n;
	return ans;
}
vector<vector<double>> Cofactor(const vector<vector<double>>& A,int row, int col)
{
	vector<vector<double>>ans;
	for (int i = 0; i < A.size(); i++)
	{
		if (i != row)
		{
			vector<double>row_elememt;
			for (int j = 0; j < A[1].size(); j++)
			{
				if (j != col)
				{
					row_elememt.push_back(A[i][j]);
				}
			}
			ans.push_back(row_elememt);
		}
	}
	return ans;
}
double calculate_det(const vector<vector<double>>& A)
{
	if (A.size() == 1)
	{
		return A[0][0];
	}
	double ans = 0;
	for (int col = 0, sign = 1; col < A[0].size(); col++, sign *= -1)
	{
		vector<vector<double>>cofactor = Cofactor(A, 0, col);
		ans += sign * A[0][col] * calculate_det(cofactor);

	}
	return ans;
}
vector<vector<double>> calculate_mat_minus_vec(const vector<vector<double>>& A, const vector<double>a)
{
	vector<vector<double>>ans;
	for (int i = 0; i < A.size(); i++)
	{
		vector<double>point;
		for (int j = 0; j < A[0].size(); j++)
		{
			point.push_back(A[i][j] - a[j]);
		}
		ans.push_back(point);
	}
	return ans;
}
vector<vector<double>> calculate_mat_add_vec(const vector<vector<double>>& A, const vector<double>a)
{
	vector<vector<double>>ans;
	for (int i = 0; i < A.size(); i++)
	{
		vector<double>point;
		for (int j = 0; j < A[0].size(); j++)
		{
			point.push_back(A[i][j] + a[j]);
		}
		ans.push_back(point);
	}
	return ans;
}
vector<vector<double>> trans(const vector<vector<double>>& A)
{
	vector<vector<double>>A_t;
	for (int i = 0; i < A[0].size(); i++)
	{
		vector<double>line;
		for (int j = 0; j < A.size(); j++)
		{
			line.push_back(A[j][i]);
		}
		A_t.push_back(line);
	}
	return A_t;
}
vector<vector<double>> mat_mul(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	vector<vector<double>>ans;
	for (int i = 0; i < A.size(); i++)
	{
		vector<double>temp;
		for (int j = 0; j < B[0].size(); j++)
		{
			double element = 0;
			for (int k = 0; k < A[0].size(); k++)
			{
				element += A[i][k] * B[k][j];
			}

			temp.push_back(element);
		}
		ans.push_back(temp);
	}
	return ans;
}
vector<double> mat_vec_mul(const vector<vector<double>>& A, const vector<double>p)
{
	vector<double>ans;
	for (int i = 0; i < A.size(); i++)
	{
		double sum = 0;
		for (int j = 0; j < A[0].size(); j++)
		{
			sum += A[i][j] * p[j];
		}
		ans.push_back(sum);
	}
	return ans;
}
vector<double> vec_minus(const vector<double>& A, const vector<double>& B)
{
	vector<double>ans;
	for (int i = 0; i < A.size(); i++)
	{
		ans.push_back(A[i] - B[i]);
	}
	return ans;
}
vector<double> vec_add(const vector<double>& A, const vector<double>& B)
{
	vector<double>ans;
	for (int i = 0; i < A.size(); i++)
	{
		ans.push_back(A[i] + B[i]);
	}
	return ans;
}
vector<int> argsort(const vector<double>& A)
{
	multimap<double, int>index;
	for (int i = 0; i < A.size(); i++)
	{
		index.insert({ A[i],i });
	}
	vector<int>ans;
	for (multimap<double, int>::iterator it = index.begin(); it != index.end(); it++)
	{
		ans.push_back(it->second);
	}
	return ans;
}
void eigen(vector<vector<double>> A, vector<vector<double>>& U, vector<double>& e)
{
	int n = A.size();
	int p = 0;
	int q = 0;
	int iter_max_num = 10000;
	int iter_num = 0;
	double eps = 1e-40;
	double max = eps;
	U.resize(n);
	e.resize(n);
	for (int i = 0; i < n; i++) {
		U[i].resize(n, 0);
		U[i][i] = 1;
	}

	while (iter_num < iter_max_num && max >= eps)
	{
		max = fabs(A[0][1]);
		p = 0;
		q = 1;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i != j && fabs(A[i][j]) > max) {
					max = fabs(A[i][j]);
					p = i;
					q = j;
				}
			}
		}

		double theta = 0.5 * atan2(2 * A[p][q], A[p][p] - A[q][q]);

		double app = A[p][p];
		double aqq = A[q][q];
		double apq = A[p][q];

		double sin_theta = sin(theta);
		double cos_theta = cos(theta);
		double sin_2theta = sin(2 * theta);
		double cos_2theta = cos(2 * theta);

		A[p][p] = app * cos_theta * cos_theta + aqq * sin_theta * sin_theta + 2 * apq * cos_theta * sin_theta;
		A[q][q] = app * sin_theta * sin_theta + aqq * cos_theta * cos_theta - 2 * apq * cos_theta * sin_theta;
		A[p][q] = 0.5 * (aqq - app) * sin_2theta + apq * cos_2theta;
		A[q][p] = A[p][q];
		for (int i = 0; i < n; i++)
		{
			if (i != p && i != q)
			{
				double Api = A[p][i];
				double Aqi = A[q][i];
				A[p][i] = Api * cos_theta + Aqi * sin_theta;
				A[i][p] = A[p][i];
				A[q][i] = -Api * sin_theta + Aqi * cos_theta;
				A[i][q] = A[q][i];
			}
		}

		for (int i = 0; i < n; i++) {
			double Uip = U[i][p];
			double Uiq = U[i][q];
			U[i][p] = Uip * cos_theta + Uiq * sin_theta;
			U[i][q] = Uiq * cos_theta - Uip * sin_theta;
		}

		iter_num++;
	}
	for (int i = 0; i < n; i++) {
		e[i] = A[i][i];
	}
	vector<int> sort_index;
	sort_index = argsort(e);
	vector<vector<double>> U_sorted(n);
	for (int i = 0; i < n; i++) {
		U_sorted[i].resize(n, 0);
	}
	vector<double> e_sorted(n);
	for (int i = 0; i < n; i++) {
		e_sorted[i] = e[sort_index[i]];
		for (int j = 0; j < n; j++) {
			U_sorted[i][j] = U[i][sort_index[j]];
		}
	}
	U = U_sorted;
	e = e_sorted;


}
vector<vector<double>> change(const vector<vector<double>>& A, const vector<vector<double>>& R, const vector<double>& t)
{
	vector<vector<double>>ans;
	for (int i = 0; i < A.size(); i++)
	{
		vector<double>point = vec_add(mat_vec_mul(R, A[i]), t);
		ans.push_back(point);
	}
	return ans;
}
vector<double> calculate_cross(const vector<double>& A, const vector<double>& B)
{
	double i = A[1] * B[2] - A[2] * B[1];
	double j = A[2] * B[0] - A[0] * B[2];
	double k = A[0] * B[1] - A[1] * B[0];
	vector<double>res = { i,j,k };
	return res;
}
double calculate_dot(const vector<double>& A, const vector<double>& B)
{
	return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
}
double calculate_norm(const vector<double>& A)
{
	return sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
}
double if_coplane(const vector<double>& A, const vector<double>& B, const vector<double>& C, const vector<double>& D)
{
	vector<double>AB = vec_minus(B, A);
	vector<double>AC = vec_minus(C, A);
	vector<double>AD = vec_minus(D, A);
	vector<double>n = calculate_cross(AB, AC);
	return fabs(calculate_dot(n, AD)/calculate_norm(n));
}
double distance(const vector<double>& P1, const vector<double>& P2, const vector<double>& Q1, const vector<double>& Q2)
{
	vector<double>t1 = { P1[0] - P2[0],P1[1] - P2[1] ,P1[2] - P2[2] };
	vector<double>t2 = { Q1[0] - Q2[0],Q1[1] - Q2[1] ,Q1[2] - Q2[2] };
	vector<double>MN = { P1[0] - Q1[0],P1[1] - Q1[1] ,P1[2] - Q1[2] };
	vector<double>n = calculate_cross(t1, t2);
	return fabs(calculate_dot(MN, n)) / calculate_norm(n);
}
vector<vector<double>>calculate_adjoint(const vector<vector<double>>& A)
{
	vector<vector<double>>res;
	res.resize(A.size());
	for (int i = 0; i < res.size(); i++)
	{
		res[i].resize(A[0].size(), 0);
	}
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A[0].size(); j++)
		{
			res[i][j]=pow(-1,i+j)*calculate_det(Cofactor(A, i, j));
		}
	}
	return res;
}
vector<vector<double>>calculate_inverse(const vector<vector<double>>& A)
{
	vector<vector<double>>res;
	res.resize(A.size());
	for (int i = 0; i < res.size(); i++)
	{
		res[i].resize(A[0].size(), 0);
	}
	double A_det = calculate_det(A);
	if (!A_det)
		return res;
	else
	{
		vector<vector<double>>A_adj=calculate_adjoint(A);
		for (int i = 0; i < A.size(); i++)
		{
			for (int j = 0; j < A[0].size(); j++)
			{
				res[i][j] = A_adj[i][j] / A_det;
			}
		}
		return res;
	}
}
vector<double> point_to_line(const vector<double>& Q, const vector<double>& P0, const vector<double>& P1)
{
	double xm = Q[0];
	double ym = Q[1];
	double zm = Q[2];
	vector<double>dir = vec_minus(P1, P0);
	double A = dir[0];
	double B = dir[1];
	double C = dir[2];
	double x0 = P0[0];
	double x1 = P1[0];
	double y0 = P0[1];
	double y1 = P1[1];
	double z0 = P0[2];
	double z1 = P1[2];
	double xn = (xm * (x1 - x0) - (y0 - ym) * (y1 - y0) + (B / A * (y1 - y0) + C / A * (z1 - z0)) * x0 - (z0 - zm) * (z1 - z0)) / (B / A * (y1 - y0) + C / A * (z1 - z0) + x1 - x0);
	double yn = y0 + B / A * (xn - x0);
	double zn = z0 + C / B * (yn - y0);
	return { xn,yn,zn };
}
vector<double> closet_point(const vector<double>& P1, const vector<double>& P2, const vector<double>& Q1, const vector<double>& Q2)
{
	double dis = distance(P1, P2, Q1, Q2);
	vector<double>t1 = vec_minus(P2, P1);
	vector<double>t2 = vec_minus(Q2, Q1);
	double A1 = t1[0];
	double B1 = t1[1];
	double C1 = t1[2];
	double A2 = t2[0];
	double B2 = t2[1];
	double C2 = t2[2];
	double x = 0;
	double y = B1 / A1 * (x - P1[0]) + P1[1];
	double z = C1 / B1 * (y - P1[1]) + P1[2];
	int turn = 1;
	int count = 0;
	vector<double>foot;
	while (1)
	{			
		if (turn == 1)
			foot = point_to_line({ x,y,z }, Q1, Q2);
		else
			foot = point_to_line({ x,y,z }, P1, P2);
		if ((fabs(calculate_distance(foot, { x,y,z }) - dis) <= 0.00001) || count == 1000000)
		{	
			break;
		}
		else
		{
			x = foot[0];
			y = foot[1];
			z = foot[2];
			count++;
			turn *= -1;
		}
	}
	return { (x + foot[0]) / 2,(y + foot[1]) / 2,(z + foot[2]) / 2 };
}
vector<double> normalize(const vector<double>& A)
{
	double norm = calculate_norm(A);
	return { A[0] / norm,A[1] / norm,A[2] / norm };
}


int main()
{
	int it_max = 100;
	int it = 0;

	vector<vector<double>>P;
	vector<vector<double>>Q;
	string line;
	//Init P
	ifstream infile("ori.txt");
	while (getline(infile, line))
	{
		istringstream str_to_num(line);
		double x, y, z;
		str_to_num >> x >> y >> z;
		vector<double>point = { x,y,z };
		P.push_back(point);
	}

	//Init Q
	ifstream infile2("tar.txt");
	while (getline(infile2, line))
	{
		istringstream str_to_num(line);
		double x, y, z;
		str_to_num >> x >> y >> z;
		vector<double>point = { x,y,z };
		Q.push_back(point);
	}

	cudaSetDevice(0);



	double** Q_host = (double**)malloc(Q.size() * sizeof(double*));
	for (int i = 0; i < Q.size(); i++)
	{
		double* Q_rowData_device;
		cudaMalloc((void**)&Q_rowData_device, Q[0].size() * sizeof(double));
		cudaMemcpy(Q_rowData_device, &Q[i][0], Q[0].size() * sizeof(double), cudaMemcpyHostToDevice);
		Q_host[i] = Q_rowData_device;
	}
	double** Q_device;
	cudaMalloc((void**)&Q_device, Q.size() * sizeof(double*));
	cudaMemcpy(Q_device, Q_host, Q.size() * sizeof(double*), cudaMemcpyHostToDevice);

	double** newQ_host = (double**)malloc(Q.size() * sizeof(double*));
	for (int i = 0; i < Q.size(); i++)
	{
		double* newQ_rowData_device;
		cudaMalloc((void**)&newQ_rowData_device, Q[0].size() * sizeof(double));
		newQ_host[i] = newQ_rowData_device;
	}
	double** newQ_device;
	cudaMalloc((void**)&newQ_device, Q.size() * sizeof(double*));
	cudaMemcpy(newQ_device, newQ_host, Q.size() * sizeof(double*), cudaMemcpyHostToDevice);

	double min_distance= std::numeric_limits<double>::max();
	int num_points = P.size();

	
	//⬇4PCS
	//4 points in P
	srand((unsigned)time(NULL));
	int a = rand() % num_points;
	int b = rand() % num_points;
	int c = rand() % num_points;
	vector<double>P1 = P[a];
	vector<double>P2 = P[b];
	vector<double>P3 = P[c];
	int min_idx = 0;
	while (min_idx == a || min_idx == b || min_idx == c)
		min_idx++;
	double min_plane = if_coplane(P1, P2, P3, P[min_idx]);
	for (int i = 0; i < num_points;i++)
	{
		if (i != a && i != b && i != c)
		{
			double temp = if_coplane(P1, P2, P3, P[i]);
			double P1Pi = calculate_distance(P1, P[i]);
			double P2Pi = calculate_distance(P2, P[i]);
			double P3Pi = calculate_distance(P3, P[i]);
			if (temp < min_plane)
			{
				min_idx = i;
				min_plane = temp;
			}
		}
	}
	vector<double>P_e;

	P_e= closet_point(P[a], P[b], P[c], P[min_idx]);
	double restrain_1_R1 = calculate_norm(vec_minus(P[a], P[b]));
	double restrain_1_R2 = calculate_norm(vec_minus(P[c], P[min_idx]));
	double restrain_2_R1 = calculate_norm(vec_minus(P[a], P_e)) / restrain_1_R1;
	double restrain_2_R2 = calculate_norm(vec_minus(P[c], P_e)) / restrain_1_R2;

	vector<pair<vector<double>, vector<double>>>points_R1;
	vector<pair<vector<double>, vector<double>>>points_R2;
	
	int** R1_set1_host = (int**)malloc(sizeof(int*) * num_points);
	int** R1_set1 = (int**)malloc(sizeof(int*) * num_points);
	for (int i = 0; i < num_points; i++)
	{
		int* pair_point;
		cudaMalloc((void**)&pair_point, sizeof(int) * 2);
		int* row = (int*)malloc(sizeof(int) * 2);
		R1_set1_host[i] = pair_point;
		R1_set1[i] = row;
	}
	int** R1_set1_device;
	cudaMalloc((void**)&R1_set1_device, sizeof(int*)* num_points);
	cudaMemcpy(R1_set1_device, R1_set1_host, sizeof(int*)* num_points, cudaMemcpyHostToDevice);
	int** R1_set2_host = (int**)malloc(sizeof(int*) * num_points);
	int** R1_set2 = (int**)malloc(sizeof(int*) * num_points);
	for (int i = 0; i < num_points; i++)
	{
		int* pair_point;
		cudaMalloc((void**)&pair_point, sizeof(int) * 2);
		int* row = (int*)malloc(sizeof(int) * 2);
		R1_set2_host[i] = pair_point;
		R1_set2[i] = row;
	}
	int** R1_set2_device;
	cudaMalloc((void**)&R1_set2_device, sizeof(int*) * num_points);
	cudaMemcpy(R1_set2_device, R1_set2_host, sizeof(int*) * num_points, cudaMemcpyHostToDevice);
	unsigned int* R1_set1_idx_device;
	unsigned int* R1_set2_idx_device;
	cudaMalloc((void**)&R1_set1_idx_device, sizeof(unsigned int));
	cudaMalloc((void**)&R1_set2_idx_device, sizeof(unsigned int));
	unsigned int R1_set1_idx_host = 0;
	unsigned int R1_set2_idx_host = 0;
	cudaMemcpy(R1_set1_idx_device, &R1_set1_idx_host, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(R1_set2_idx_device, &R1_set2_idx_host, sizeof(unsigned int), cudaMemcpyHostToDevice);

	int blocksize_points_set_R1 = 1024;
	int gridsize_points_set_R1 = ceil((float)(num_points * num_points) / (float)blocksize_points_set_R1);
	find_points_R1<<<gridsize_points_set_R1, blocksize_points_set_R1 >>> (Q_device, R1_set1_device, R1_set2_device, restrain_1_R1, restrain_1_R2, R1_set1_idx_device, R1_set2_idx_device, num_points);

	point_e* e_set1_device;
	cudaMalloc((void**)&e_set1_device, sizeof(point_e) * num_points);
	point_e* e_set2_device;
	cudaMalloc((void**)&e_set2_device, sizeof(point_e) * num_points);

	int blocksize_calculate_e = 1024;
	int gridsize_calculate_e = ceil((float)(num_points) / (float)blocksize_calculate_e);
	find_e << <gridsize_calculate_e, blocksize_calculate_e >> > (Q_device, R1_set1_device, restrain_2_R1, e_set1_device, num_points);
	find_e << <gridsize_calculate_e, blocksize_calculate_e >> > (Q_device, R1_set2_device, restrain_2_R2, e_set2_device, num_points);

	int** final_set_host = (int**)malloc(sizeof(int*) * num_points);
	for (int i = 0; i < num_points; i++)
	{
		int* row;
		cudaMalloc((void**)&row, sizeof(int) * 4);
		int temp[4] = { -1,-1,-1,-1 };
		cudaMemcpy(row, &temp[0], sizeof(int) * 4, cudaMemcpyHostToDevice);
		final_set_host[i] = row;
	}
	int** final_set_device;
	cudaMalloc((void**)&final_set_device, sizeof(int*) * num_points);
	cudaMemcpy(final_set_device, final_set_host, sizeof(int*) * num_points, cudaMemcpyHostToDevice);
	int final_set_count_host = -1;
	int* final_set_count_device;
	cudaMalloc((void**)&final_set_count_device, sizeof(int));
	cudaMemcpy(final_set_count_device, &final_set_count_host, sizeof(int),cudaMemcpyHostToDevice);

	double** point_e_host = (double**)malloc(sizeof(double*) * num_points);
	for (int i = 0; i < num_points; i++)
	{
		double* temp;
		cudaMalloc((void**)&temp, sizeof(double) * 3);
		point_e_host[i] = temp;
	}
	double** point_e_device;
	cudaMalloc((void**)&point_e_device, sizeof(double*)* num_points);
	cudaMemcpy(point_e_device, point_e_host, sizeof(double*)* num_points, cudaMemcpyHostToDevice);

	int blocksize_final_set = 1024;
	int gridsize_final_set = ceil((float)(num_points * num_points) / (float)blocksize_final_set);
	final_four_points_set <<<gridsize_final_set, blocksize_final_set >>> (e_set1_device, e_set2_device, final_set_device,point_e_device,final_set_count_device, num_points);
	cudaMemcpy(&final_set_count_host, final_set_count_device, sizeof(int), cudaMemcpyDeviceToHost);
	int** final_set = (int**)malloc(sizeof(int*) * final_set_count_host);
	double** final_set_e = (double**)malloc(sizeof(double*) * final_set_count_host);
	for (int i = 0; i < final_set_count_host; i++)
	{
		int* temp = (int*)malloc(sizeof(int) * 4);
		cudaMemcpy(temp, final_set_host[i], sizeof(int) * 4, cudaMemcpyDeviceToHost);
		final_set[i] = temp;

		double* temp_e = (double*)malloc(sizeof(double) * 3);
		cudaMemcpy(temp_e, point_e_host[i], sizeof(double) * 3, cudaMemcpyDeviceToHost);
		final_set_e[i] = temp_e;
	}

	vector<double>best_t = { 0,0,0 };
	vector<vector<double>>best_rotate = { {1,0,0},{0,1,0},{0,0,1} };
	double best_match = 0;
	for (int i = 0; i < final_set_count_host; i++)
	{
		printf("now: %d tatal: %d\n", i+1, final_set_count_host);
		vector<double>t = { final_set_e[i][0] - P_e[0] ,final_set_e[i][1] - P_e[1] ,final_set_e[i][2] - P_e[2] };
		vector<double>vb = normalize(calculate_cross(vec_minus(P[b],P[a]), vec_minus(P[c],P[a])));
		int A_idx = final_set[i][0];
		int B_idx = final_set[i][1];
		int C_idx = final_set[i][2];
		int D_idx = final_set[i][3];
		vector<double>va = normalize(calculate_cross(vec_minus(Q[B_idx], Q[A_idx]), vec_minus(Q[C_idx], Q[A_idx])));
		vector<double>vs = calculate_cross(vb, va);
		vector<double>v = normalize(vs);
		double ca = calculate_dot(vb, va);
		vector<double>vt = { v[0] * (1 - ca),v[1] * (1 - ca),v[2] * (1 - ca) };
		vector<vector<double>>rm;
		rm.resize(3);
		for (int i = 0; i < 3; i++)
		{
			rm[i].resize(3, 0);
			rm[i][i] = 1;
		}
		rm[0][0] = vt[0] * v[0] + ca;
		rm[1][1] = vt[1] * v[1] + ca;
		rm[2][2] = vt[2] * v[2] + ca;
		vt[0] *= v[1];
		vt[2] *= v[0];
		vt[1] *= v[2];
		rm[0][1] = vt[0] - vs[2];
		rm[0][2] = vt[2] + vs[1];
		rm[1][0] = vt[0] + vs[2];
		rm[1][2] = vt[1] - vs[0];
		rm[2][0] = vt[2] - vs[1];
		rm[2][1] = vt[1] + vs[0];

		vector<vector<double>>newP;
		newP.resize(num_points);
		int temp_best_match = 0;
		for (int j = 0; j < num_points; j++)
		{
			newP[j] = vec_add(mat_vec_mul(rm, P[j]),t);
			if (calculate_distance(newP[j], Q[j]) < 0.1)
			{
				temp_best_match += 1;
			}
		}

		if (temp_best_match > best_match)
		{
			best_t = t;
			for (int j = 0; j < 3; j++)
			{
				best_rotate[j] = rm[j];
			}
		}
		
	}
	for (int i = 0; i < num_points; i++)
	{
		P[i] = vec_add(mat_vec_mul(best_rotate, P[i]), best_t);
	}
	ofstream outfile("4pcsresult.txt");
	for (int i = 0; i < P.size(); i++)
	{
		for (int j = 0; j < P[0].size(); j++)
		{
			outfile << P[i][j] << " ";
		}
		outfile << endl;
	}
	
	cudaFree((void*)R1_set1_idx_device);
	cudaFree((void*)R1_set2_idx_device);
	cudaFree((void*)e_set1_device);
	cudaFree((void*)e_set2_device);
	cudaFree((void*)point_e_device);
	cudaFree((void*)final_set_count_device);

	//⬆ 4PCS

	double** P_host = (double**)malloc(P.size() * sizeof(double*));
	for (int i = 0; i < P.size(); i++)
	{
		double* P_rowData_device;
		cudaMalloc((void**)&P_rowData_device, P[0].size() * sizeof(double));
		cudaMemcpy(P_rowData_device, &P[i][0], P[0].size() * sizeof(double), cudaMemcpyHostToDevice);
		P_host[i] = P_rowData_device;
	}
	double** P_device;
	cudaMalloc((void**)&P_device, P.size() * sizeof(double*));
	cudaMemcpy(P_device, P_host, P.size() * sizeof(double*), cudaMemcpyHostToDevice);

	
	//ICP
	while (1)
	{
		clock_t start = clock();

		//Init Q'
		int blocksize_initnewQ = 1024;
		int gridsize_initnewQ = ceil((float)(num_points) / (float)blocksize_initnewQ);
		Init_new_Q << <gridsize_initnewQ, blocksize_initnewQ >> > (P_device, Q_device, newQ_device, num_points);

		//Calculate center
		int blocksize_calculate_center = 1024;
		int gridsize_calculate_center = ceil((float)num_points / (float)blocksize_calculate_center);

		double* p_x_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* p_y_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* p_z_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* p_x_center_device;
		double* p_y_center_device;
		double* p_z_center_device;
		cudaMalloc((void**)&p_x_center_device, gridsize_calculate_center * sizeof(double));
		cudaMalloc((void**)&p_y_center_device, gridsize_calculate_center * sizeof(double));
		cudaMalloc((void**)&p_z_center_device, gridsize_calculate_center * sizeof(double));
		calculate_center << <gridsize_calculate_center, blocksize_calculate_center >> > (P_device, p_x_center_device, p_y_center_device, p_z_center_device, num_points);
		cudaMemcpy(p_x_center_host, p_x_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_y_center_host, p_y_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_z_center_host, p_z_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		double p_center[3] = { 0,0,0 };
		for (int i = 0; i < gridsize_calculate_center; i++)
		{
			p_center[0] += p_x_center_host[i];
			p_center[1] += p_y_center_host[i];
			p_center[2] += p_z_center_host[i];
		}
		double* p_center_device;
		cudaMalloc((void**)&p_center_device, 3 * sizeof(double));
		cudaMemcpy(p_center_device, p_center, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaFree((void*)p_x_center_device);
		cudaFree((void*)p_y_center_device);
		cudaFree((void*)p_z_center_device);

		double* newq_x_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* newq_y_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* newq_z_center_host = (double*)malloc(sizeof(double) * gridsize_calculate_center);
		double* newq_x_center_device;
		double* newq_y_center_device;
		double* newq_z_center_device;
		cudaMalloc((void**)&newq_x_center_device, gridsize_calculate_center * sizeof(double));
		cudaMalloc((void**)&newq_y_center_device, gridsize_calculate_center * sizeof(double));
		cudaMalloc((void**)&newq_z_center_device, gridsize_calculate_center * sizeof(double));
		calculate_center << <gridsize_calculate_center, blocksize_calculate_center >> > (newQ_device, newq_x_center_device, newq_y_center_device, newq_z_center_device, num_points);
		cudaMemcpy(newq_x_center_host, newq_x_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		cudaMemcpy(newq_y_center_host, newq_y_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		cudaMemcpy(newq_z_center_host, newq_z_center_device, sizeof(double) * gridsize_calculate_center, cudaMemcpyDeviceToHost);
		double newq_center[3] = { 0,0,0 };
		for (int i = 0; i < gridsize_calculate_center; i++)
		{
			newq_center[0] += newq_x_center_host[i];
			newq_center[1] += newq_y_center_host[i];
			newq_center[2] += newq_z_center_host[i];
		}
		double* newq_center_device;
		cudaMalloc((void**)&newq_center_device, 3 * sizeof(double));
		cudaMemcpy(newq_center_device, newq_center, 3 * sizeof(double), cudaMemcpyHostToDevice);
		cudaFree((void*)newq_x_center_device);
		cudaFree((void*)newq_y_center_device);
		cudaFree((void*)newq_z_center_device);

		//Mat minus center
		int blocksize_mat_minus_vec = 1024;
		int gridsize_mat_minus_vec = ceil((float)num_points / (float)blocksize_mat_minus_vec);
		mat_minus_vec << <gridsize_mat_minus_vec, blocksize_mat_minus_vec >> > (P_device, p_center_device, num_points);
		mat_minus_vec << <gridsize_mat_minus_vec, blocksize_mat_minus_vec >> > (newQ_device, newq_center_device, num_points);
		cudaFree((void*)newq_center_device);

		//Init S
		double S[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
		double** S_host = (double**)malloc(3 * sizeof(double*));
		for (int i = 0; i < 3; i++)
		{
			double* S_rowData_device;
			cudaMalloc((void**)&S_rowData_device, 3 * sizeof(double));
			cudaMemcpy(S_rowData_device, &S[i][0], 3 * sizeof(double), cudaMemcpyHostToDevice);
			S_host[i] = S_rowData_device;
		}
		double** S_device;
		cudaMalloc((void**)&S_device, 3 * sizeof(double*));
		cudaMemcpy(S_device, S_host, 3 * sizeof(double*), cudaMemcpyHostToDevice);
		int blocksize_initS = 1024;
		int gridsize_initS = ceil((float)num_points * 9 / (float)blocksize_initS);
		mat_mul << <gridsize_initS, blocksize_initS >> > (P_device, newQ_device, S_device, num_points, num_points * 9, true);
		mat_add_vec << <gridsize_mat_minus_vec, blocksize_mat_minus_vec >> > (P_device, p_center_device, num_points);
		cudaFree((void*)p_center_device);


		
		//SVD
		for (int i = 0; i < 3; i++)
		{
			cudaMemcpy(S[i], S_host[i], 3 * sizeof(double), cudaMemcpyDeviceToHost);
		}
		cudaFree((void**)S_device);
		vector<vector<double>>S_v = { {0,0,0},{0,0,0},{0,0,0} };
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				S_v[i][j] = S[i][j];
		}
		vector<vector<double>>E;
		vector<double>e;
		vector<vector<double>>S_t = trans(S_v);
		vector<vector<double>>STS = mat_mul(S_t, S_v);

		eigen(STS, E, e);
		vector<vector<double>>sigma_inverse, V;
		sigma_inverse.resize(e.size());
		V.resize(e.size());
		for (int i = 0; i < e.size(); i++)
		{
			sigma_inverse[i].resize(e.size(), 0);
			V[i].resize(e.size(), 0);
		}
		for (int i = 0; i < E.size(); i++)
		{
			for (int j = E[0].size() - 1, k = 0; j >= 0; j--, k++)
			{
				V[i][k] = E[i][j];
			}
		}

		for (int i = 0; i < e.size(); i++)
		{
			for (int j = 0; j < e.size(); j++)
			{
				if (i == j)
				{
					sigma_inverse[i][j] = 1/sqrt(e[e.size()-1-i]);
				}
			}
		}
		vector<vector<double>>U = mat_mul(S_v, mat_mul(V, sigma_inverse));
		vector<vector<double>>U_t = trans(U);

		//calculate R
		vector<vector<double>>R = mat_mul(V, trans(U));
		if (calculate_det(R)+1 <1e-10 )
		{
			vector<vector<double>>temp;
			temp.resize(U_t.size());
			for (int i = 0; i < temp.size(); i++)
			{
				temp[i].resize(U_t[0].size(), 0);
				if (i != temp.size() - 1)
				{
					temp[i][i] = 1;
				}
				else
				{
					temp[i][i] = -1;
				}
			}
			R = mat_mul(V, mat_mul(temp, U_t));
			printf("%lf\n", calculate_det(R));
		}

		double** R_host = (double**)malloc(3 * sizeof(double*));
		for (int i = 0; i < 3; i++)
		{
			double* R_rowData_device;
			cudaMalloc((void**)&R_rowData_device, 3 * sizeof(double));
			cudaMemcpy(R_rowData_device, &R[i][0], 3 * sizeof(double), cudaMemcpyHostToDevice);
			R_host[i] = R_rowData_device;
		}
		double** R_device;
		cudaMalloc((void**)&R_device, 3 * sizeof(double*));
		cudaMemcpy(R_device, R_host, 3 * sizeof(double*), cudaMemcpyHostToDevice);

		//calculate t
		vector<double>p = { 0,0,0 };
		vector<double>newq = { 0,0,0 };
		for (int i = 0; i < 3; i++)
		{
			p[i] = p_center[i];
			newq[i] = newq_center[i];
		}
		vector<double>rp = mat_vec_mul(R, p);
		vector<double>t = vec_minus(newq, rp);
		double* t_device;
		cudaMalloc((void**)&t_device, 3 * sizeof(double));
		cudaMemcpy(t_device, &t[0], 3 * sizeof(double), cudaMemcpyHostToDevice);

		//calculate P-Q distance
		int blocksize_calculate_distance = 1024;
		int gridsize_calculate_distance = ceil((float)num_points / (float)blocksize_calculate_distance);
		double* distance_block_host = (double*)malloc(sizeof(double) * gridsize_calculate_distance);
		double* distance_block_device;
		cudaMalloc((void**)&distance_block_device, sizeof(double) * gridsize_calculate_distance);
		calculate_distance << <gridsize_calculate_distance, blocksize_calculate_distance >> > (P_device, Q_device, num_points, distance_block_device);
		cudaMemcpy(distance_block_host, distance_block_device, sizeof(double) * gridsize_calculate_distance, cudaMemcpyDeviceToHost);
		double newdistance = 0;
		for (int i = 0; i < gridsize_calculate_distance; i++)
		{
			newdistance += distance_block_host[i];
		}

		//calculate Q_next
		int blocksize_calculate_Qnext = 1024;
		int gridsize_calculate_Qnext = ceil((float)num_points / (float)blocksize_calculate_Qnext);
		change << <gridsize_calculate_Qnext, blocksize_calculate_Qnext >> > (R_device, P_device, t_device, num_points);
		cudaFree((void*)t_device);
		cudaFree((void*)distance_block_device);
		cudaFree((void*)R_device);
		it++;
		clock_t end = clock();
		cout << "it: " << it << " distance: " << newdistance << " in " << (double)(end - start)/CLOCKS_PER_SEC<<" seconds. " << endl;
		if (newdistance < min_distance)
		{
			vector<vector<double>>result;
			result.resize(num_points);
			for (int i = 0; i < result.size(); i++)
			{
				result[i].resize(3);
				cudaMemcpy(&result[i][0], P_host[i], 3 * sizeof(double), cudaMemcpyDeviceToHost);
			}
			ofstream outfile("result.txt");
			for (int i = 0; i < result.size(); i++)
			{
				for (int j = 0; j < result[0].size(); j++)
				{
					outfile << result[i][j] << " ";
				}
				outfile << endl;
			}
			min_distance = newdistance;
		}
		if (newdistance < 1e-40 || it == it_max)
		{
			break;
		}
	}
	
	
	return 0;
}