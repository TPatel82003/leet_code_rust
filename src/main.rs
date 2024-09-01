struct Solution;
use std::cmp::{max, min, Ordering};
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::i32::{MAX, MIN};
use std::vec;
pub struct CallData {
    id: i32,
    timestamp: i32,
}
#[allow(dead_code)]
impl Solution {
    pub fn min_processing_time(mut processor_time: Vec<i32>, mut tasks: Vec<i32>) -> i32 {
        processor_time.sort();
        tasks.sort_by(|a, b| b.cmp(&a));
        let mut ans: i32 = MAX;
        for i in 0..processor_time.len() {
            ans = min(ans, processor_time[i] + tasks[i * 4]);
        }
        ans
    }
    pub fn min_rectangles_to_cover_points(mut points: Vec<Vec<i32>>, w: i32) -> i32 {
        points.sort_by(|a, b| a[0].cmp(&b[0]));
        let x_points: Vec<i32> = points.iter().map(|coordinates| coordinates[0]).collect();
        let mut i = 0;
        let mut ans: i32 = 0;
        while i < x_points.len() {
            let l = x_points[i];
            while i < x_points.len() && x_points[i] - l <= w {
                i += 1;
            }
            ans += 1;
        }
        ans
    }
    pub fn number_of_subarrays(nums: Vec<i32>, k: i32) -> i32 {
        let mut ans = 0;
        let mut count: HashMap<i32, i32> = HashMap::new();
        let mut odd_count: i32 = 0;
        for &num in &nums {
            odd_count += num & 1;
            ans += count.get(&(odd_count - k)).unwrap_or(&0) + (odd_count == k) as i32;
            *count.entry(odd_count).or_default() += 1;
        }
        ans
    }
    pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, minutes: i32) -> i32 {
        let mut ans: i32 = MIN;
        let mut prefix: Vec<i32> = Vec::from(customers.clone());
        for i in 1..prefix.len() {
            prefix[i] += prefix[i - 1];
        }
        let mut with_grumpy: Vec<i32> = vec![0; customers.len()];
        with_grumpy[0] = match grumpy[0] == 1 {
            true => 0,
            false => customers[0],
        };
        for i in 1..customers.len() {
            with_grumpy[i] = if grumpy[i] == 1 { 0 } else { customers[i] } + with_grumpy[i - 1];
        }
        let mut i: usize = 0;
        while i + minutes as usize <= customers.len() {
            let use_power = prefix[i + minutes as usize - 1]
                - match i.gt(&0) {
                    true => prefix[i - 1],
                    false => 0,
                };
            ans = max(
                ans,
                use_power + *with_grumpy.iter().last().unwrap()
                    - with_grumpy[i + minutes as usize - 1]
                    + match i.gt(&0) {
                        true => with_grumpy[i - 1],
                        false => 0,
                    },
            );
            i += 1;
        }
        ans
    }
    fn folder_vector(input: Vec<i32>) -> Vec<i32> {
        // Return a new vector containing the elements from the input vector,
        // except for every two consecutive elements that are equal.
        let mut result: Vec<_> = input
            .windows(2)
            .filter(|window| window[0] != window[1])
            .map(|window| window[0])
            .collect();
        if let Some(&[first, second]) = input.windows(2).last() {
            result.push(if first == second { first } else { second });
        }
        result
    }
    pub fn get_first_last_from_map(window: BTreeMap<i32, i32>) -> (i32, i32) {
        let l = window.last_key_value().unwrap();
        let r = window.first_key_value().unwrap();
        (*l.0, *r.0)
    }
    pub fn longest_subarray(nums: Vec<i32>, limit: i32) -> i32 {
        let mut ans: i32 = 0;
        let mut window: BTreeMap<i32, i32> = BTreeMap::new();
        let mut l = 0;
        for r in 0..nums.len() {
            *window.entry(nums[r]).or_insert(0) += 1;
            let (mut smallest, mut largest) = Solution::get_first_last_from_map(window.clone());
            match largest - smallest <= limit {
                true => {
                    ans = max(ans, (r - l + 1) as i32);
                }
                false => {
                    while l <= r && (largest - smallest) > limit {
                        *window.get_mut(&nums[l]).unwrap() -= 1;
                        if *window.get_mut(&nums[l]).unwrap() == 0 {
                            window.remove(&nums[l]);
                        }
                        l += 1;
                        (smallest, largest) = Solution::get_first_last_from_map(window.clone());
                    }
                }
            }
        }
        ans
    }
    pub fn maximum_total_cost(nums: Vec<i32>) -> i64 {
        let n = nums.len();
        if n <= 1 {
            return nums[0] as i64;
        }
        let mut dp: Vec<Vec<i64>> = vec![vec![0; 2]; n];
        dp[1][0] = (nums[0] + nums[1]) as i64;
        dp[1][1] = (nums[0] - nums[1]) as i64;
        for i in 2..n {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]) + nums[i] as i64;
            dp[i][1] = dp[i - 1][0] + nums[i] as i64;
        }
        return max(dp[n - 1][0], dp[n - 1][1]);
    }
    pub fn minimum_area(grid: Vec<Vec<i32>>) -> i32 {
        let mut dimensions: Vec<i32> = vec![MAX, MIN, MAX, MIN];
        let get_index = |k: usize, i: usize, j: usize| match k <= 1 {
            true => i,
            false => j,
        } as i32;
        for i in 0..grid.len() {
            for j in 0..grid[0].len() {
                if grid[i][j] == 1 {
                    for k in 0..4 {
                        dimensions[k] = match k == 0 || k == 2 {
                            true => min(dimensions[k], get_index(k, i, j)),
                            false => max(dimensions[k], get_index(k, i, j)),
                        }
                    }
                }
            }
        }
        return (dimensions[1] - dimensions[0] + 1) * (dimensions[3] - dimensions[2] + 1);
    }
    pub fn find_maximum_sequence(nums: &[i32], mut start: i32) -> usize {
        let mut count = 0;
        for &num in nums {
            if num == start {
                count += 1;
                start = !start;
            }
        }
        count
    }
    pub fn maximum_length(mut nums: Vec<i32>) -> i32 {
        let count_10 = Solution::find_maximum_sequence(&nums, 1) as i32;
        let count_01 = Solution::find_maximum_sequence(&nums, 0) as i32;
        let n = nums.len();
        for i in 0..n {
            nums[i] %= 2;
        }
        let count_z = nums
            .into_iter()
            .fold(0, |acc, x| if x == 0 { acc + 1 } else { acc });
        let count_o = n as i32 - count_z;
        return max(max(count_01, count_10), max(count_o, count_z));
    }
    pub fn exclusive_time(n: i32, logs: Vec<String>) -> Vec<i32> {
        let mut execution_time: Vec<i32> = vec![0; n as usize];
        let mut call_stack: Vec<CallData> = vec![];
        for log in logs {
            let data = log.split(':').collect::<Vec<&str>>();
            let (id, timestamp) = match (data[0].parse::<i32>().ok(), data[2].parse::<i32>().ok()) {
                (Some(id), Some(timestamp)) => (id, timestamp),
                (_, _) => (0, 0),
            };
            if call_stack.is_empty() || data[1] == "start" {
                call_stack.push(CallData {
                    id: id,
                    timestamp: timestamp,
                })
            } else {
                if let Some(data) = call_stack.last_mut() {
                    let calculated_duration = timestamp - data.timestamp + 1;
                    execution_time[data.id as usize] += calculated_duration;
                    call_stack.pop();
                    if let Some(recursive) = call_stack.last_mut() {
                        execution_time[recursive.id as usize] -= calculated_duration;
                    }
                }
            }
        }
        execution_time
    }
    pub fn num_water_bottles(mut num_bottles: i32, num_exchange: i32) -> i32 {
        let mut drink = 0;
        let mut extra = 0;
        while num_bottles > 0 {
            let reminder = num_bottles % num_exchange;
            let current = num_bottles - reminder;
            drink += current;
            if num_bottles < num_exchange {
                break;
            }
            extra += current / num_exchange;
            num_bottles = extra + reminder;
        }
        drink + extra
    }
    pub fn get_number_of_backlog_orders(orders: Vec<Vec<i32>>) -> i32 {
        let mut ans = 0;
        let mut buys_freq: BTreeMap<i32, i32> = BTreeMap::new();
        let mut sell_freq: BTreeMap<i32, i32> = BTreeMap::new();
        const MOD: i32 = 1e9 as i32 + 7;
        for j in orders {
            let (price, mut amount, o_type) = (j[0], j[1], j[2]);
            match o_type {
                0 => {
                    while let Some((min_sells, min_sells_freq)) =
                        sell_freq.clone().first_key_value()
                    {
                        if *min_sells > price || amount <= 0 {
                            break;
                        }
                        let min_freq = min(*min_sells_freq, amount);
                        amount -= min_freq;
                        *sell_freq.entry(*min_sells).or_insert(0) -= min_freq;
                        if *min_sells_freq <= min_freq {
                            sell_freq.remove_entry(min_sells);
                        }
                    }
                    if amount > 0 {
                        *buys_freq.entry(price).or_insert(0) += amount;
                    }
                }
                _ => {
                    while let Some((min_sells, min_sells_freq)) = buys_freq.clone().last_key_value()
                    {
                        if *min_sells < price || amount <= 0 {
                            break;
                        }
                        let min_freq = min(*min_sells_freq, amount);
                        amount -= min_freq;
                        *buys_freq.entry(*min_sells).or_insert(0) -= min_freq;
                        if *min_sells_freq <= min_freq {
                            buys_freq.remove_entry(min_sells);
                        }
                    }
                    if amount > 0 {
                        *sell_freq.entry(price).or_insert(0) += amount;
                    }
                }
            }
        }
        for (_i, j) in buys_freq {
            ans = (ans + j % MOD) % MOD;
        }
        for (_i, j) in sell_freq {
            ans = (ans + j % MOD) % MOD;
        }
        ans
    }
    pub fn get_verify_binary(str: String) -> bool {
        let binary_vec: Vec<char> = str.chars().collect();
        binary_vec
            .windows(2)
            .all(|items| items[0] == '1' || items[1] == '1')
    }
    pub fn valid_strings(n: i32) -> Vec<String> {
        let mut ans = vec![];
        for i in 0..(1 << n) {
            let bin_i = format!("{:0>width$}", format!("{:b}", i), width = n as usize);
            if Solution::get_verify_binary(bin_i.clone()) {
                ans.push(bin_i);
            }
        }
        ans
    }
    pub fn survived_robots_healths(
        positions: Vec<i32>,
        mut healths: Vec<i32>,
        directions: String,
    ) -> Vec<i32> {
        let n = positions.len();
        let mut ans: Vec<i32> = vec![];
        let mut index: Vec<usize> = (0..positions.len()).collect();
        let mut stack: Vec<usize> = vec![];
        let direction_vec = directions.chars().collect::<Vec<char>>();
        index.sort_by(|a, b| positions[*a].cmp(&positions[*b]));
        for current_index in index {
            if direction_vec[current_index] == 'R' {
                stack.push(current_index);
            } else {
                while !stack.is_empty() && healths[current_index] > 0 {
                    let top_index = *stack.last().unwrap_or(&0);
                    stack.pop();
                    let order = healths[top_index].cmp(&healths[current_index]);
                    healths[top_index] = match order {
                        Ordering::Equal | Ordering::Less => 0,
                        Ordering::Greater => {
                            stack.push(top_index);
                            healths[top_index] - 1
                        }
                    };
                    healths[current_index] = match order {
                        Ordering::Equal | Ordering::Greater => 0,
                        Ordering::Less => healths[current_index] - 1,
                    };
                }
            }
        }
        for i in 0..n {
            if healths[i] > 0 {
                ans.push(healths[i]);
            }
        }
        ans
    }
    pub fn restore_matrix(mut row_sum: Vec<i32>, mut col_sum: Vec<i32>) -> Vec<Vec<i32>> {
        let n = row_sum.len();
        let m = col_sum.len();
        let mut ans: Vec<Vec<i32>> = vec![vec![0; m]; n];
        for i in 0..n {
            for j in 0..m {
                ans[i][j] = min(row_sum[i], col_sum[j]);
                row_sum[i] -= ans[i][j];
                col_sum[j] -= ans[i][j];
            }
        }
        ans
    }
    pub fn max_operations(s: String) -> i32 {
        let mut ans: i32 = 0;
        let mut positions: Vec<usize> = vec![];
        let chars = s.chars().collect::<Vec<char>>();
        for i in 0..s.len() {
            if chars[i] == '1' {
                positions.push(i);
            }
        }
        for i in 1..positions.len() {
            if positions[i] - positions[i - 1] > 1 {
                ans += i as i32;
            }
        }
        if chars[chars.len() - 1] == '0' {
            ans += positions.len() as i32;
        }
        ans
    }

    pub fn bfs(start: i32, end: i32, n: &i32, adj: Vec<Vec<Node>>) -> f64 {
        let mut probability = vec![0.0; *n as usize];
        probability[start as usize] = 1.0;
        let mut priority_queue: BinaryHeap<Node> = BinaryHeap::new();
        priority_queue.push(Node {
            node: start,
            prob: 1.0,
        });
        while !priority_queue.is_empty() {
            let current = priority_queue.pop().unwrap();
            for j in adj[current.node as usize].iter() {
                if probability[j.node as usize] < current.prob * j.prob {
                    probability[j.node as usize] = current.prob * j.prob;
                    priority_queue.push(Node {
                        node: j.node,
                        prob: probability[j.node as usize],
                    });
                }
            }
        }
        return probability[end as usize];
    }
    pub fn max_probability(
        n: i32,
        edges: Vec<Vec<i32>>,
        succ_prob: Vec<f64>,
        start_node: i32,
        end_node: i32,
    ) -> f64 {
        let mut adj: Vec<Vec<Node>> = vec![Default::default(); n as usize];
        for i in 0..edges.len() {
            adj[edges[i][0] as usize].push(Node {
                node: edges[i][1],
                prob: succ_prob[i],
            });
            adj[edges[i][1] as usize].push(Node {
                node: edges[i][0],
                prob: succ_prob[i],
            });
        }
        Solution::bfs(start_node, end_node, &n, adj)
    }
    pub fn sort_strip_number(num: i32) -> Vec<char> {
        num.to_string().chars().collect()
    }
    pub fn swap_and_check(number1: i32, number2: i32) -> bool {
        let mut number1 = Solution::sort_strip_number(number1);
        for i in 0..number1.len() {
            for j in (i + 1)..number1.len() {
                number1.swap(i, j);
                if number1.iter().collect::<String>().parse::<i32>().unwrap() == number2 {
                    return true;
                }
                number1.swap(i, j);
            }
        }
        false
    }
    pub fn count_pairs(nums: Vec<i32>) -> i32 {
        let mut ans = 0;
        for i in 0..nums.len() {
            for j in (i + 1)..nums.len() {
                if nums[i] == nums[j]
                    || Solution::swap_and_check(nums[i], nums[j])
                    || Solution::swap_and_check(nums[j], nums[i])
                {
                    ans += 1;
                }
            }
        }
        ans
    }
}
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct Node {
    node: i32,
    prob: f64,
}
impl Default for Node {
    fn default() -> Self {
        Self { node: 0, prob: 0.0 }
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.prob.partial_cmp(&self.prob).unwrap()
    }
}
fn main() {
    println!("Hello, world!");
}
#[allow(dead_code)]
fn folder_vector(input: Vec<i32>) -> Vec<i32> {
    let mut result: Vec<_> = input
        .windows(2)
        .filter(|window| window[0] != window[1])
        .map(|window| window[0])
        .collect();
    if let Some(&[first, second]) = input.windows(2).last() {
        result.push(if first == second { first } else { second });
    }
    result
}
#[cfg(test)]
mod test {
    use super::*;
    use rstest::*;
    #[rstest]
    #[case(vec![0, 1, 1, 1, 1, 0, 0, 0, 1] , vec![0, 1, 0, 1])]
    #[case(vec![0, 1, 1, 1, 1, 0, 0, 0, 1, 1] , vec![0, 1, 0, 1])]
    #[case(vec![0, 1, 1, 1, 1, 0, 0, 0, 1, 0] , vec![0, 1, 0, 1, 0])]
    fn test_folded_vector(#[case] input: Vec<i32>, #[case] expected: Vec<i32>) {
        assert_eq!(Solution::folder_vector(input), expected);
    }
    #[rstest]
    #[case(vec![1,2,5,6] , vec![10,10,11,11] , "RLRL" , vec![])]
    #[case(vec![3,5,2,6] , vec![10,10,15,12] , "RLRL" , vec![14])]
    #[case(vec![5,4,3,2,1] , vec![2,17,9,15,10] , "RRRRR" , vec![2,17,9,15,10])]
    fn test_survived_robots_healths(
        #[case] positions: Vec<i32>,
        #[case] healths: Vec<i32>,
        #[case] directions: String,
        #[case] expected: Vec<i32>,
    ) {
        assert_eq!(
            Solution::survived_robots_healths(positions, healths, directions),
            expected
        );
    }
    #[rstest]
    #[case(vec![3,8] , vec![4,7] , vec![vec![3,0],vec![1,7]])]
    #[case(vec![5,7,10] , vec![8,6,8] , vec![vec![5,0,0],vec![3,4,0],vec![0,2,8]])]

    fn test_restore_matrix(
        #[case] row_sum: Vec<i32>,
        #[case] col_sum: Vec<i32>,
        #[case] expected: Vec<Vec<i32>>,
    ) {
        assert_eq!(Solution::restore_matrix(row_sum, col_sum), expected);
    }
    #[rstest]
    #[case("00111", 0)]
    #[case("1001101", 4)]
    #[case("10101010", 10)]
    #[case("1010101010001000011101010110", 64)]
    fn test_max_operations(#[case] s: String, #[case] expected: i32) {
        assert_eq!(Solution::max_operations(s), expected);
    }
    #[rstest]
    #[case(3,
        vec![vec![0,1],vec![1,2],vec![0,2]],
        vec![0.5,0.5,0.2],
        0,
        2)]
    fn test_maximum_probability(
        #[case] n: i32,
        #[case] edges: Vec<Vec<i32>>,
        #[case] succ_prob: Vec<f64>,
        #[case] start_node: i32,
        #[case] end_node: i32,
    ) {
        assert_eq!(
            Solution::max_probability(n, edges, succ_prob, start_node, end_node),
            0.25
        );
    }
    #[rstest]
    #[case(vec![123,231] , 0)]
    #[case(vec![1,1,1,1,1] , 10)]
    #[case(vec![3,12,30,17,21] , 2)]
    fn test_count_pairs(#[case] nums: Vec<i32>, #[case] expected: i32) {
        assert_eq!(Solution::count_pairs(nums), expected);
    }
}
