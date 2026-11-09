#ifndef PIMCOMP_OPEN_GENETICALGORITHM_H
#define PIMCOMP_OPEN_GENETICALGORITHM_H
#include "../common.h"
#include "../configure.h"
#include "PIMCOMPVariable.h"
#include "../evaluation/EvaluationConfig.h"

enum SimpleOp : uint8_t
{
    SIMPLE_OP_NONE = 0,
    SIMPLE_OP_LD = 1,
    SIMPLE_OP_ST = 2,
    SIMPLE_OP_MVMUL = 3,
    SIMPLE_OP_SEND = 4,
    SIMPLE_OP_RECV = 5
};

struct SimpleInst
{
    uint8_t op = SIMPLE_OP_NONE;
    int node_index = 0;
    int AG_index = 0;
    int comm_index = 0;
    int instruction_index_in_core = 0;
    int from_core = 0; // RECV
    int to_core = 0;   // SEND
    int element_num = 0;
};

class GeneticAlgorithm
{
public:
    GeneticAlgorithm(std::string model_name_, std::string pipeline_granularity_);
    void GeneticAlgorithmOptimizing();
private:
    std::string model_name;
    std::string pipeline_granularity;
    double GA_MVMUL_start_interval;
    int core_num = ChipH * ChipW;
    int node_num = PIMCOMP_node_list_origin.size();

    int population_num = 200;
    int max_iteration = 3;

    int max_AG_kind_per_core = 32;
    int chromosome_size = max_AG_kind_per_core * core_num;

    double remain_rate = 0.3;
    double random_select_rate = 0.2;
    double increase_replication_num_rate = 0.2;
    double decrease_replication_num_rate = 0.2;
    double split_rate = 0.2;
    double gather_rate = 0.2;
    double exchange_rate = 0.2;

    int conv_num;
    int fc_num;
    std::vector<int> CONV_FC_index;
    std::vector<int> CONV_index;
    std::vector<int> FC_index;
    std::vector<int> node_crossbar_num_per_AG;
    std::vector<int> node_AG_num;
    std::vector<int> node_cycle_num;
    std::vector<int> node_MVMUL_num;
    std::vector<std::vector<int>> current_chromosome;
    std::vector<std::vector<int>> node_replication_num;
    std::vector<std::vector<int>> core_AG_type_num;
    std::vector<std::vector<int>> core_crossbar_num;
    std::vector<double> current_fitness;
    std::vector<int> fitness_scratch;
    std::unordered_map<uint64_t, double> fitness_cache;
    // Init
    void Init();
    void InitReplicating(float margin_factor, int individual_index);
    void InitMapping(int individual_index);
    void InitMappingClustered(int individual_index);
    void InitMappingDistributed(int individual_index);
    // Select
    void Select();
    double Fitness(const std::vector<int>& individual, const std::vector<int>& replication_num_vector);
    double FastEvaluationForBatch(const std::vector<int>& individual, const std::vector<int>& replication_num_vector);
    double FastEvaluationForElement(const std::vector<int>& individual, const std::vector<int>& replication_num_vector);
    double FastEvaluationInstructionGroupNum(const std::vector<int>& replication_num_vector);
    void FastSingleInstructionGroupEvaluation(int instruction_group_index, int core_index, int index_in_core);
    int CheckBusBandwidth(int chip_index, long long current_time, int communication_needed_cycle);
    int CheckGlobalMemoryBandwidth(int chip_index, long long current_time, int global_memory_needed_cycle);
    std::vector<std::vector<uint32_t>> COMM_cycle_flag;
    std::vector<std::vector<uint32_t>> MEM_cycle_flag;
    uint32_t current_comm_eval_id = 1;
    uint32_t current_mem_eval_id = 1;
    // Mutation
    void Mutation();
    bool CheckLegality(const std::vector<int> & individual, const std::vector<int> & replication_num_vector, const std::vector<int> & AG_type_num_vector, const std::vector<int> & crossbar_num_vector, const std::string& Stage);
    // Post Process
    void PostProcess();
    void SaveIntermediateInfo();
};

#endif //PIMCOMP_OPEN_GENETICALGORITHM_H
