#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


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

double calculate_distance(const vector<double>& P, const vector<double>& Q)
{
	return (P[0] - Q[0]) * (P[0] - Q[0]) + (P[1] - Q[1]) * (P[1] - Q[1]) + (P[2] - Q[2]) * (P[2] - Q[2]);
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
vector<vector<double>> Cofactor(const vector<vector<double>>& A, int col)
{
	vector<vector<double>>ans;
	for (int i = 1; i < A.size(); i++)
	{
		vector<double>row;
		for (int j = 0; j < A[1].size(); j++)
		{
			if (j != col)
			{
				row.push_back(A[i][j]);
			}
		}
		ans.push_back(row);
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
		vector<vector<double>>cofactor = Cofactor(A, col);
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
	ifstream infile2("tar2.txt");
	while (getline(infile2, line))
	{
		istringstream str_to_num(line);
		double x, y, z;
		str_to_num >> x >> y >> z;
		vector<double>point = { x,y,z };
		Q.push_back(point);
	}

	cudaSetDevice(0);

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