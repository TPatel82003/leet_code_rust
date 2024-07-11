struct Solution;

use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap};

use std::i32::{MAX, MIN};
use std::vec;
pub struct CallData {
    id: i32,
    timestamp: i32,
}
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
    pub fn min_operations(nums: Vec<i32>) -> i32 {
        if nums.len() < 2 {
            return if nums[0] == 1 { 0 } else { 1 };
        }
        let mut ans: i32 = 0;
        let folded: Vec<i32> = Solution::folder_vector(nums);
        ans = folded.iter().fold(0, |acc, x| acc + (x == &0) as i32) * 2;
        ans -= (*folded.iter().last().unwrap() == 0) as i32;
        ans
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
                    while let Some((min_sells, min_sells_freq)) = buys_freq.clone().last_key_value() {
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
}
fn main() {
    let binding = vec![0, 1, 1, 1, 1, 0, 0, 0, 1, 1];
    println!("{:?} -> {:?}", binding, folder_vector(binding.clone()))
}
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
    #[test]
    fn test_fold() {
        assert_eq!(
            folder_vector(vec![0, 1, 1, 1, 1, 0, 0, 0, 1]),
            vec![0, 1, 0, 1]
        );
        assert_eq!(
            folder_vector(vec![0, 1, 1, 1, 1, 0, 0, 0, 1, 1]),
            vec![0, 1, 0, 1]
        );
        assert_eq!(
            folder_vector(vec![0, 1, 1, 1, 1, 0, 0, 0, 1, 0]),
            vec![0, 1, 0, 1, 0]
        );
        assert_eq!(folder_vector(vec![0, 0, 0]), vec![0]);
        let mut b: BTreeMap<i32, i32> = BTreeMap::new();
        b.insert(3, 5);
        b.insert(1, 5);
        b.insert(4, 5);
        let top = b.first_entry().unwrap();
        assert_eq!(top.key(), &2);
    }
}
