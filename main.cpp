#include <bits/stdc++.h>

using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right)
      : val(x), left(left), right(right) {}
};

class Node {
 public:
  int val;
  vector<Node *> children;
  Node() {}
  Node(int _val) { val = _val; }
  Node(int _val, vector<Node *> _children) {
    val = _val;
    children = _children;
  }
};

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// 中序遍历
class Solution_94 {
 public:
  vector<int> inorderTraversal(TreeNode *root) {
    if (root == nullptr) return {};

    vector<int> ans;
    stack<TreeNode *> tree;
    TreeNode *curr = root;
    while (tree.empty() == false || curr != nullptr) {
      while (curr != nullptr) {
        tree.push(curr);
        curr = curr->left;
      }
      curr = tree.top();
      tree.pop();
      ans.push_back(curr->val);
      curr = curr->right;
    }
    return ans;
  }
};

// 前序遍历
class Solution_144 {
 public:
  vector<int> preorderTraversal(TreeNode *root) {
    if (root == nullptr) return {};

    vector<int> ans;
    stack<TreeNode *> s;

    s.push(root);

    while (s.empty() == false) {
      TreeNode *tmp = s.top();
      s.pop();
      ans.push_back(tmp->val);
      if (tmp->right != nullptr) s.push(tmp->right);
      if (tmp->left != nullptr) s.push(tmp->left);
    }

    return ans;
  }
};

// 后序遍历
class Solution_145 {
 public:
  vector<int> postorderTraversal(TreeNode *root) {
    if (root == nullptr) return {};
    vector<int> ans;
    stack<TreeNode *> s;

    TreeNode *curr = root, *last_visited = nullptr;
    while (s.empty() == false || curr != nullptr) {
      while (curr != nullptr) {
        s.push(curr);
        curr = curr->left;
      }

      curr = s.top();
      if (curr->right == nullptr || last_visited == curr->right) {
        ans.push_back(curr->val);
        last_visited = curr;
        s.pop();
        curr = nullptr;
      } else {
        curr = curr->right;
      }
    }
    return ans;
  }
};

// 层序遍历
class Solution_429 {
 public:
  vector<vector<int>> levelOrder(Node *root) {
    if (root == nullptr) return {};

    vector<vector<int>> ans;
    queue<Node *> q;
    q.push(root);
    while (q.empty() == false) {
      int n = q.size();
      ans.push_back({});
      for (int i = 0; i < n; ++i) {
        ans.back().push_back(q.front()->val);
        for_each(q.front()->children.begin(), q.front()->children.end(),
                 [&](Node *i) { q.push(i); });
        q.pop();
      }
    }
    return ans;
  }
};

// N叉树前序遍历
class Solution_589 {
 public:
  vector<int> preorder(Node *root) {
    if (root == nullptr) return {};
    vector<int> ans;
    stack<Node *> s;
    s.push(root);
    while (s.empty() == false) {
      Node *tmp = s.top();
      s.pop();
      ans.push_back(tmp->val);

      for_each(tmp->children.rbegin(), tmp->children.rend(),
               [&](Node *i) { s.push(i); });
    }

    return ans;
  }
};

// N叉树后序遍历
class Solution_590 {
 public:
  vector<int> postorder(Node *root) {
    if (root == nullptr) return {};
    vector<int> ans;

    stack<Node *> s;
    s.push(root);
    while (s.empty() == false) {
      Node *tmp = s.top();
      s.pop();
      ans.push_back(tmp->val);
      for_each(tmp->children.begin(), tmp->children.end(),
               [&](Node *i) { s.push(i); });
    }
    reverse(ans.begin(), ans.end());
    return ans;
  }
};

// TODO: 二叉树垂序遍历
class Solution_987 {
 public:
  vector<vector<int>> verticalTraversal(TreeNode *root) {
    if (root == nullptr) return {};
    vector<vector<int>> ans;

    return ans;
  }
};

// 层数最深叶子节点的和
class Solution_1302 {
 public:
  int deepestLeavesSum(TreeNode *root) {
    if (root == nullptr) return 0;
    int ans = 0;

    queue<TreeNode *> q;
    q.push(root);
    while (q.empty() == false) {
      int n = q.size();
      ans = 0;
      for (int i = 0; i < n; ++i) {
        ans += q.front()->val;
        if (q.front()->left != nullptr) q.push(q.front()->left);
        if (q.front()->right != nullptr) q.push(q.front()->right);
        q.pop();
      }
    }

    return ans;
  }
};

// 相同的树
class Solution_100 {
 public:
  bool isSameTree(TreeNode *p, TreeNode *q) {
    if (p == nullptr && q == nullptr) return true;
    if (p == nullptr || q == nullptr) return false;
    if (p->val != q->val) return false;

    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
  }
};

// 对称二叉树
class Solution_101 {
 public:
  bool isMirror(TreeNode *p, TreeNode *q) {
    if (p == nullptr && q == nullptr) return true;
    if (p == nullptr || q == nullptr) return false;

    if (p->val != q->val) return false;
    return isMirror(p->left, q->right) && isMirror(p->right, q->left);
  }

  bool isSymmetric(TreeNode *root) { return isMirror(root, root); }
};

// 二叉树的最大深度
class Solution_104 {
 public:
  int maxDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    int l = maxDepth(root->left);
    int r = maxDepth(root->right);
    return max(l, r) + 1;
  }
};

// 平衡二叉树
class Solution_110 {
 public:
  map<TreeNode *, int> m;

  int depth(TreeNode *root) {
    if (root == nullptr) return 0;
    if (m.count(root) != 0) return m[root];
    int l = m.count(root->left) == 0 ? depth(root->left) : m[root->left];
    int r = m.count(root->right) == 0 ? depth(root->right) : m[root->right];
    return m[root] = max(l, r) + 1;
  }

  bool isBalanced(TreeNode *root) {
    if (root == nullptr) return true;
    return abs(depth(root->left) - depth(root->right)) < 2 &&
           isBalanced(root->left) && isBalanced(root->right);
  }
};

// 二叉树的最小深度
class Solution_111 {
 public:
  int minDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    if (root->left == nullptr && root->right == nullptr) return 1;
    int l = root->left == nullptr ? INT_MAX : minDepth(root->left);
    int r = root->right == nullptr ? INT_MAX : minDepth(root->right);
    return min(l, r) + 1;
  }
};

// 另一棵树的子树
class Solution_572 {
 public:
  bool isSame(TreeNode *p, TreeNode *q) {
    if (p == nullptr && q == nullptr) return true;
    if (q == nullptr || p == nullptr) return false;
    if (p->val != q->val) return false;
    return isSame(p->left, q->left) && isSame(p->right, q->right);
  }

  bool isSubtree(TreeNode *root, TreeNode *subRoot) {
    if (root == nullptr && subRoot != nullptr) return false;
    return isSame(root, subRoot) || isSubtree(root->left, subRoot) ||
           isSubtree(root->right, subRoot);
  }
};

// 单值二叉树
class Solution_965 {
 public:
  bool isUnivalTree(TreeNode *root) {
    if (root == nullptr) return true;
    return (root->val ==
                (root->left == nullptr ? root->val : root->left->val) &&
            root->val ==
                (root->right == nullptr ? root->val : root->right->val)) &&
           isUnivalTree(root->left) && isUnivalTree(root->right);
  }
};

// 814.二叉树剪枝
class Solution_814 {
 public:
  TreeNode *pruneTree(TreeNode *root) {
    if (root == nullptr) return nullptr;
    root->left = pruneTree(root->left);
    root->right = pruneTree(root->right);
    return (root->val == 0 && root->left == nullptr && root->right == nullptr)
               ? nullptr
               : root;
  }
};

// 669. 修剪二叉搜索树
class Solution_669 {
 public:
  TreeNode *trimBST(TreeNode *root, int low, int high) {
    if (root == nullptr) return nullptr;

    if (root->val < low) return trimBST(root->right, low, high);
    if (root->val > high) return trimBST(root->left, low, high);

    root->left = trimBST(root->left, low, high);
    root->right = trimBST(root->right, low, high);

    return root;
  }
};

// 1325. 删除给定值的叶子节点
class Solution_1325 {
 public:
  TreeNode *removeLeafNodes(TreeNode *root, int target) {
    if (root == nullptr) return nullptr;

    root->left = removeLeafNodes(root->left, target);
    root->right = removeLeafNodes(root->right, target);

    if (root->val == target && root->left == nullptr && root->right == nullptr)
      return nullptr;

    return root;
  }
};

// 112. 路径总和
class Solution_112 {
 public:
  bool hasPathSum(TreeNode *root, int targetSum) {
    if (root == nullptr) return false;
    if (root->left == nullptr && root->right == nullptr)
      return root->val == targetSum;

    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
  }
};

// 113. 路径总和 II
class Solution_113 {
 private:
  vector<vector<int>> ans;

  void dfs(TreeNode *root, vector<int> &v, int target) {
    if (root == nullptr) return;
    if (root->left == nullptr && root->right == nullptr) {
      if (root->val == target) {
        ans.push_back(v);
        ans.back().push_back(root->val);
      }
      return;
    }

    v.push_back(root->val);
    dfs(root->left, v, target - root->val);
    dfs(root->right, v, target - root->val);
    v.pop_back();
  }

 public:
  vector<vector<int>> pathSum(TreeNode *root, int targetSum) {
    vector<int> curr;
    dfs(root, curr, targetSum);
    return ans;
  }
};

// 437. 路径总和 III
class Solution_437 {
 private:
  int ans = 0;
  unordered_map<int, int> m;

  void dfs(TreeNode *root, int prefix_sum, int target) {
    if (root == nullptr) return;

    prefix_sum += root->val;
    ans += m[prefix_sum - target];
    ++m[prefix_sum];
    dfs(root->left, prefix_sum, target);
    dfs(root->right, prefix_sum, target);
    --m[prefix_sum];
  }

 private:
 public:
  int pathSum(TreeNode *root, int targetSum) {
    m.insert({0, 1});
    dfs(root, 0, targetSum);
    return ans;
  }
};

// 129. 求根节点到叶节点数字之和
class Solution_129 {
 private:
  int ans;

  void dfs(TreeNode *root, int curr) {
    if (root == nullptr) return;
    if (root->left == nullptr && root->right == nullptr) {
      ans += curr * 10 + root->val;
      return;
    }

    dfs(root->left, curr * 10 + root->val);
    dfs(root->right, curr * 10 + root->val);
  }

 public:
  int sumNumbers(TreeNode *root) {
    dfs(root, 0);
    return ans;
  }
};

// 257. 二叉树的所有路径
class Solution_257 {
 private:
  vector<string> ans;

  void dfs(TreeNode *root, string &curr) {
    if (root == nullptr) return;
    if (root->left == nullptr && root->right == nullptr) {
      ans.push_back(curr);
      ans.back() += to_string(root->val);
      return;
    }

    string tmp = curr + to_string(root->val) + "->";
    dfs(root->left, tmp);
    dfs(root->right, tmp);
  }

 public:
  vector<string> binaryTreePaths(TreeNode *root) {
    if (root == nullptr) return {};
    string curr;
    dfs(root, curr);
    return ans;
  }
};

// TODO: 297. 二叉树的序列化与反序列化
class Codec_297 {
 public:
  // Encodes a tree to a single string.
  string serialize(TreeNode *root) { return {}; }

  // Decodes your encoded data to tree.
  TreeNode *deserialize(string data) { return {}; }
};

// TODO: 449. 序列化和反序列化二叉搜索树
class Codec_449 {
 public:
  // Encodes a tree to a single string.
  string serialize(TreeNode *root) { return {}; }

  // Decodes your encoded data to tree.
  TreeNode *deserialize(string data) { return {}; }
};

// 508. 出现次数最多的子树元素和
class Solution_508 {
 private:
  int max_frequency = -1;
  vector<int> ans;
  unordered_map<int, int> m;

  int tree_sum(TreeNode *root) {
    if (root == nullptr) return 0;

    int sum = root->val + tree_sum(root->left) + tree_sum(root->right);
    int freq = ++m[sum];
    if (freq > max_frequency) {
      max_frequency = freq;
      ans.clear();
    }
    if (freq == max_frequency) ans.push_back(sum);
    return sum;
  }

 public:
  vector<int> findFrequentTreeSum(TreeNode *root) {
    tree_sum(root);
    return ans;
  }
};

// TODO: 124. 二叉树中的最大路径和
class Solution_124 {
 private:
  int ans = 0;

  int max_sum(TreeNode *root) {
    if (root == nullptr) return 0;
    int l = max(0, max_sum(root->left)), r = max(0, max_sum(root->right));
    int sum = l + r + root->val;
    ans = max(ans, sum);
    return max(l, r) + root->val;
  }

 public:
  int maxPathSum(TreeNode *root) {
    ans = INT_MIN;
    max_sum(root);
    return ans;
  }
};

// 543. 二叉树的直径
class Solution_543 {
 private:
  int ans = 0;

  int diameter_tree(TreeNode *root) {
    if (root == nullptr) return -1;
    int l = diameter_tree(root->left) + 1, r = diameter_tree(root->right) + 1;
    ans = max(ans, l + r);
    return max(l, r);
  }

 public:
  int diameterOfBinaryTree(TreeNode *root) {
    ans = 0;
    diameter_tree(root);
    return ans;
  }
};

// TODO: ???? 687. 最长同值路径
class Solution_687 {
 private:
  int ans = 0;

  int dfs(TreeNode *root) {
    if (root == nullptr) return 0;
    int l = dfs(root->left), r = dfs(root->right), pl = 0, pr = 0;
    if (root->left != nullptr && root->val == root->left->val) pl = l + 1;
    if (root->right != nullptr && root->val == root->right->val) pr = r + 1;
    ans = max(ans, pl + pr);
    return max(pl, pr);
  }

 public:
  int longestUnivaluePath(TreeNode *root) {
    dfs(root);
    return ans;
  }
};

// 133. 克隆图
class Solution_133 {
 private:
  unordered_map<Node *, Node *> m;

 public:
  Node *cloneGraph(Node *node) {
    if (node == nullptr) return nullptr;
    if (m.count(node)) return m[node];

    Node *new_node = new Node(node->val);
    m[node] = new_node;

    for (auto &n : node->children) {
      new_node->children.push_back(cloneGraph(n));
    }

    return new_node;
  }
};

// 138. 复制带随机指针的链表
class Solution {
  class Node {
   public:
    int val;
    Node *next;
    Node *random;

    Node(int _val) {
      val = _val;
      next = NULL;
      random = NULL;
    }
  };

 private:
  unordered_map<Node *, Node *> m;

 public:
  Node *copyRandomList(Node *head) {
    if (head == nullptr) return nullptr;
    if (m.count(head)) return m[head];

    Node *new_node = new Node(head->val);
    m[head] = new_node;

    new_node->next = copyRandomList(head->next);
    new_node->random = copyRandomList(head->random);
    return new_node;
  }
};

// 200. 岛屿数量
class Solution_200 {
 private:
  int col = 0, row = 0;
  vector<int> dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};

  void dfs(vector<vector<char>> &g, int _row, int _col) {
    if (_col < 0 || _row < 0 || _col >= col || _row >= row ||
        g[_row][_col] == '0')
      return;

    g[_row][_col] = '0';

    for (int i = 0; i < 4; ++i) {
      dfs(g, _row + dx[i], _col + dy[i]);
    }
  }

 public:
  int numIslands(vector<vector<char>> &grid) {
    row = grid.size();
    if (row == 0) return 0;
    col = grid.front().size();

    int ans = 0;

    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        ans += grid[i][j] - '0';
        dfs(grid, i, j);
      }
    }
    return ans;
  }
};

// 547. 省份数量
class Solution_547 {
 private:
  int row = 0;

  void dfs(vector<vector<int>> &g, int _row) {
    for (int i = 0; i < row; ++i) {
      if (g[_row][i] == 0) continue;

      g[_row][i] = 0;
      dfs(g, i);
    }
  }

 public:
  int findCircleNum(vector<vector<int>> &isConnected) {
    if (isConnected.empty() == true) return 0;
    row = isConnected.size();

    int ans = 0;

    for (int i = 0; i < row; ++i) {
      ans += isConnected[i][i];
      dfs(isConnected, i);
    }

    return ans;
  }
};

// 695. 岛屿的最大面积
class Solution_695 {
 private:
  int ans = 0;
  int _row = 0, _col = 0;
  vector<int> dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};

  void dfs(vector<vector<int>> &g, int row, int col, int &curr) {
    if (row < 0 || col < 0 || row >= _row || col >= _col || g[row][col] == 0) {
      ans = max(ans, curr);
      return;
    }

    g[row][col] = 0;
    ++curr;

    for (int i = 0; i < 4; ++i) {
      dfs(g, row + dx[i], col + dy[i], curr);
    }
  }

 public:
  int maxAreaOfIsland(vector<vector<int>> &grid) {
    if (grid.empty() == true) return 0;

    _row = grid.size();
    _col = grid.front().size();

    for (int i = 0; i < _row; ++i) {
      for (int j = 0; j < _col; ++j) {
        if (grid[i][j] == 0) continue;
        int curr = 0;
        dfs(grid, i, j, curr);
      }
    }

    return ans;
  }
};

// ----------------------- 9.8 -----------------------

// 733. 图像渲染
class Solution_733 {
 private:
  int _row, _col;
  vector<int> dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};

  void dfs(vector<vector<int>> &g, int row, int col, int new_color,
           int old_color) {
    if (row < 0 || col < 0 || row >= _row || col >= _col ||
        g[row][col] != old_color)
      return;

    g[row][col] = new_color;

    for (int i = 0; i < 4; ++i) {
      dfs(g, row + dx[i], col + dy[i], new_color, old_color);
    }
  }

 public:
  vector<vector<int>> floodFill(vector<vector<int>> &image, int sr, int sc,
                                int newColor) {
    if (image.empty() == true || newColor == image[sr][sc]) return image;
    _row = image.size(), _col = image.front().size();
    dfs(image, sr, sc, newColor, image[sr][sc]);
    return image;
  }
};

// TODO: 827. 最大人工岛
class Solution_827 {
 private:
  int ans = 0;

 public:
  int largestIsland(vector<vector<int>> &grid) { return ans; }
};

// 1162. 地图分析
class Solution_1162 {
 private:
  const vector<int> dx = {0, 1, 0, -1}, dy = {1, 0, -1, 0};

 public:
  int maxDistance(vector<vector<int>> &grid) {
    int ans = -1;
    if (grid.empty() == true) return ans;

    const int row = grid.size(), col = grid.front().size();
    queue<pair<int, int>> q;
    vector<vector<bool>> v(row, vector<bool>(col, false));

    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        if (grid[i][j] == 1) {
          q.emplace(i, j);
          v[i][j] = true;
        }
      }
    }

    int step = 0;
    while (q.empty() == false) {
      for (int n = q.size(); n > 0; --n) {
        int x = q.front().first, y = q.front().second;
        q.pop();

        if (v[x][y] == true && grid[x][y] == 0) ans = max(ans, step);

        for (int i = 0; i < 4; ++i) {
          int nx = x + dx[i], ny = y + dy[i];
          if (nx < 0 || ny < 0 || nx >= row || ny >= col || v[nx][ny] == true)
            continue;

          q.emplace(nx, ny);
          v[nx][ny] = true;
        }
      }
      ++step;
    }

    return ans;
  }
};

// 841. 钥匙和房间
class Solution_841 {
 private:
  unordered_set<int> ans;
  void dfs(vector<vector<int>> &rooms, int curr) {
    if (ans.count(curr)) return;
    ans.insert(curr);

    for (auto &r : rooms[curr]) {
      dfs(rooms, r);
    }
  }

 public:
  bool canVisitAllRooms(vector<vector<int>> &rooms) {
    dfs(rooms, 0);
    return ans.size() == rooms.size();
  }
};

// ----------------------- 9.9 -----------------------

// TODO: 1202. 交换字符串中的元素
class Solution_1202 {
 public:
  string smallestStringWithSwaps(string s, vector<vector<int>> &pairs) {
    vector<vector<int>> g(s.length());
    for (auto &p : pairs) {
      g[p[0]].push_back(p[1]);
      g[p[1]].push_back(p[0]);
    }

    unordered_set<int> v;
    vector<int> index;
    string tmp;

    function<void(int)> dfs = [&](int curr) {
      if (v.count(curr)) return;
      v.insert(curr);
      index.push_back(curr);
      tmp += s[curr];
      for (auto nxt : g[curr]) dfs(nxt);
    };

    for (int i = 0; i < s.length(); ++i) {
      if (v.count(i)) continue;

      index.clear();
      tmp.clear();

      dfs(i);

      sort(tmp.begin(), tmp.end());
      sort(index.begin(), index.end());

      for (int j = 0; j < index.size(); ++j) {
        s[index[j]] = tmp[j];
      }
    }
    return s;
  }
};

// 207. 课程表
class Solution_207 {
 private:
  unordered_map<int, unordered_set<int>> m;

  bool hasCycle(int start, int curr, vector<bool> &v) {
    if (start == curr && v[start] == true) return true;

    if (m.count(curr) == 0) return false;

    for (const auto nxt : m[curr]) {
      if (v[nxt] == true) continue;
      v[nxt] = true;
      if (hasCycle(start, nxt, v) == true) return true;
    }
    return false;
  }

 public:
  bool canFinish(int numCourses, vector<vector<int>> &prerequisites) {
    for (auto &p : prerequisites) {
      m[p.front()].insert(p.back());
    }

    for (int i = 0; i < numCourses; ++i) {
      vector<bool> v(numCourses, false);
      if (hasCycle(i, i, v) == true) return false;
    }

    return true;
  }
};

// 210. 课程表 II
class Solution_210 {
 private:
  vector<int> ans;
  unordered_map<int, unordered_set<int>> m;

  bool hasCycle(int curr, vector<int> &v) {
    if (v[curr] == 1) return true;
    if (v[curr] == 2) return false;

    v[curr] = 1;

    for (auto nxt : m[curr]) {
      if (hasCycle(nxt, v)) return true;
    }

    v[curr] = 2;
    ans.push_back(curr);

    return false;
  }

 public:
  vector<int> findOrder(int numCourses, vector<vector<int>> &prerequisites) {
    for (auto &p : prerequisites) {
      m[p.front()].insert(p.back());
    }

    vector<int> v(numCourses, 0);

    for (int i = 0; i < numCourses; ++i) {
      if (hasCycle(i, v)) return {};
    }
    //    std::reverse(ans.begin(), ans.end());
    return ans;
  }
};

// 802. 找到最终的安全状态
class Solution_802 {
 private:
  vector<int> ans;

  bool dfs(vector<vector<int>> &g, int curr, vector<int> &v) {
    if (v[curr] == 1) return true;
    if (v[curr] == 2) return false;

    v[curr] = 1;

    for (auto nxt : g[curr]) {
      if (dfs(g, nxt, v)) return true;
    }
    v[curr] = 2;
    return false;
  }

 public:
  vector<int> eventualSafeNodes(vector<vector<int>> &graph) {
    vector<int> v(graph.size(), 0);
    for (int i = 0; i < graph.size(); ++i) {
      if (dfs(graph, i, v)) continue;
      ans.push_back(i);
    }
    sort(ans.begin(), ans.end());
    return ans;
  }
};

// ----------------------- 9.12 -----------------------

// 399. 除法求值
class Solution_399 {
 private:
  unordered_map<string, unordered_map<string, double>> m;
  vector<double> ans;

  double dfs(string curr, string end, unordered_set<string> &v) {
    if (curr == end) return 1.0;
    v.insert(curr);

    for (auto &p : m[curr]) {
      if (v.count(p.first) != 0) continue;

      double ret = dfs(p.first, end, v);
      if (ret > 0) return ret * p.second;
    }

    return -1.0;
  }

 public:
  vector<double> calcEquation(vector<vector<string>> &equations,
                              vector<double> &values,
                              vector<vector<string>> &queries) {
    for (int i = 0; i < equations.size(); ++i) {
      m[equations[i].front()].emplace(equations[i].back(), values[i]);
      m[equations[i].back()].emplace(equations[i].front(), 1 / values[i]);
    }

    for (auto &querie : queries) {
      string a = querie.front(), b = querie.back();
      if (m.count(a) == 0 || m.count(b) == 0) {
        ans.emplace_back(-1.0);
        continue;
      }

      unordered_set<string> v;

      ans.emplace_back(dfs(a, b, v));
    }

    return ans;
  }
};

// ----------------------- 9.13 -----------------------

// TODO: 839. 相似字符串组
class Solution_839 {
 private:
  int ans = 0;
  unordered_set<string> s;

  int matching(string str) {
    int n = str.size();
    int ret = 0;
    unordered_set<string> v;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        string tmp = str;
        tmp[i] = str[j];
        tmp[j] = str[i];

        if (v.count(tmp) != 0) continue;

        v.insert(tmp);

        if (s.count(tmp) != 0) ret += 1;
      }
    }
    return ret;
  }

 public:
  int numSimilarGroups(vector<string> &strs) {
    if (strs.empty() == true) return 0;
    s.insert(strs.front());

    for (int i = 1; i < strs.size(); ++i) {
      string str = strs[i];
      ans += matching(str);
      s.insert(str);
    }

    return ans;
  }
};

// TODO: 952. 按公因数计算最大组件大小
class Solution_952 {
 public:
  int largestComponentSize(vector<int> &nums) {
    int ans = 0;

    return ans;
  }
};

// 990. 等式方程的可满足性
class Solution_990 {
 private:
 public:
  bool equationsPossible(vector<string> &equations) {
    bool ans = false;
    return ans;
  }
};

// 721. 账户合并
class Solution_721 {
 public:
  vector<vector<string>> accountsMerge(vector<vector<string>> &accounts) {
    unordered_map<string_view, int> ids;

    return {};
  }
};

// TODO: 785. 判断二分图
class Solution_785 {
 private:
  vector<int> colors;
  int n = 0;

  bool coloring(const vector<vector<int>> &graph, int color, int node) {
    if (colors[node] != 0) return colors[node] == color;

    colors[node] = color;
    for (int nxt : graph[node]) {
      if (coloring(graph, -color, nxt) == false) return false;
    }
    return true;
  }

 public:
  bool isBipartite(vector<vector<int>> &graph) {
    n = graph.size();
    colors.resize(n);

    for (int i = 0; i < n; ++i) {
      if (colors[i] == 0 && coloring(graph, 1, i) == false) return false;
    }
    return true;
  }
};

// 886. 可能的二分法
class Solution_886 {
 private:
  vector<int> colors;
  unordered_map<int, vector<int>> g;

  bool coloring(int node, int color) {
    if (colors[node] != 0) return colors[node] == color;

    colors[node] = color;

    for (int nxt : g[node])
      if (coloring(nxt, -color) == false) return false;

    return true;
  }

 public:
  bool possibleBipartition(int n, vector<vector<int>> &dislikes) {
    colors = vector<int>(n, 0);

    for (auto &item : dislikes) {
      g[item.front() - 1].emplace_back(item.back() - 1);
      g[item.back() - 1].emplace_back(item.front() - 1);
    }

    for (int i = 0; i < n; ++i)
      if (colors[i] == 0 && coloring(i, 1) == false) return false;

    return true;
  }
};

// TODO: 1042. 不邻接植花
class Solution_1042 {
 private:
 public:
  vector<int> gardenNoAdj(int n, vector<vector<int>> &paths) {
    unordered_map<int, vector<int>> m;
    for (const auto &path : paths) {
      m[path.front() - 1].emplace_back(path.back() - 1);
      m[path.back() - 1].emplace_back(path.front() - 1);
    }

    vector<int> ans(n, 0);

    for (int i = 0; i < n; ++i) {
      int mask = 0;
      for (const auto &j : m[i]) {
        mask |= (1 << ans[j]);
      }

      for (int c = 1; c <= 4 && ans[i] == 0; ++c) {
        if (!(mask & (1 << c))) ans[i] = c;
      }
    }

    return ans;
  }
};

// 997. 找到小镇的法官
class Solution_997 {
 public:
  int findJudge(int n, vector<vector<int>> &trust) {
    vector<int> ans(n + 1, 0);

    for (const auto &i : trust) {
      --ans[i.front()];
      ++ans[i.back()];
    }

    for (int i = 1; i <= n; ++i) {
      if (ans[i] == (n - 1)) return i;
    }

    return -1;
  }
};

// ----------------------- 9.22 -----------------------

// 433. 最小基因变化
class Solution_433 {
 private:
  bool isValid(const string &s, const string &e) {
    int count = 0;
    for (int i = 0; i < s.size(); ++i) {
      if (s[i] != e[i] && count++) return false;
    }

    return true;
  }

 public:
  int minMutation(string start, string end, vector<string> &bank) {
    queue<string> q;
    q.push(start);

    unordered_set<string> v;

    int ans = 0;
    while (q.empty() == false) {
      for (int n = q.size(); n > 0; --n) {
        string curr = std::move(q.front());
        q.pop();

        if (curr == end) return ans;

        for (const string &item : bank) {
          if (v.count(item) != 0 || isValid(item, curr) == false) continue;

          v.insert(item);
          q.push(item);
        }
      }
      ++ans;
    }

    return -1;
  }
};

// 815. 公交路线
class Solution_815 {
 public:
  int numBusesToDestination(vector<vector<int>> &routes, int source,
                            int target) {
    if (source == target) return 0;
    unordered_map<int, vector<int>> m;

    for (int i = 0; i < routes.size(); ++i) {
      for (int j = 0; j < routes[i].size(); ++j) {
        m[routes[i][j]].push_back(i);
      }
    }

    unordered_set<int> v;
    queue<int> q;
    for (int index : m[source]) {
      q.push(index);
    }

    int ans = 1;

    while (q.empty() == false) {
      for (int n = q.size(); n > 0; --n) {
        int curr = q.front();
        q.pop();
        for (auto station : routes[curr]) {
          if (station == target) return ans;
          for (auto index : m[station]) {
            if (v.count(index) != 0) continue;
            v.insert(index);
            q.push(index);
          }
        }
      }
      ++ans;
    }
    return -1;
  }
};

// ----------------------- 9.23 -----------------------

// 863. 二叉树中所有距离为 K 的结点
class Solution_863 {
 private:
  unordered_map<TreeNode *, vector<TreeNode *>> g;

  void buildGraph(TreeNode *parent, TreeNode *child) {
    if (parent != nullptr) {
      g[parent].push_back(child);
      g[child].push_back(parent);
    }

    if (child->left != nullptr) buildGraph(child, child->left);
    if (child->right != nullptr) buildGraph(child, child->right);
  }

 public:
  vector<int> distanceK(TreeNode *root, TreeNode *target, int k) {
    if (root == nullptr) return {};
    buildGraph(nullptr, root);
    vector<int> ans;
    unordered_set<TreeNode *> v;
    v.insert(target);
    queue<TreeNode *> q;
    q.push(target);
    int step = 0;
    while (q.empty() == false && step <= k) {
      for (int n = q.size(); n > 0; --n) {
        TreeNode *curr = q.front();
        q.pop();

        if (step == k) ans.push_back(curr->val);

        for (auto child : g[curr]) {
          if (v.count(child) != 0) continue;

          q.push(child);
          v.insert(child);
        }
      }
      ++step;
    }
    return ans;
  }
};

// ----------------------- 9.28 -----------------------

// 1129. 颜色交替的最短路径
class Solution_1129 {
 private:
  enum ColorType { red, blue };

 public:
  vector<int> shortestAlternatingPaths(int n, vector<vector<int>> &red_edges,
                                       vector<vector<int>> &blue_edges) {
    unordered_map<int, unordered_set<int>> edges_r, edges_b;
    for (auto &e : red_edges) edges_r[e[0]].insert(e[1]);
    for (auto &e : blue_edges) edges_b[e[0]].insert(e[1]);

    unordered_set<int> seen_r, seen_b;
    seen_r.insert(0);
    seen_b.insert(0);

    queue<pair<int, ColorType>> q;
    q.emplace(0, ColorType::red);
    q.emplace(0, ColorType::blue);

    vector<int> ans(n, -1);
    int step = 0;
    while (q.empty() == false) {
      for (int size_q = q.size(); size_q > 0; --size_q) {
        int curr = q.front().first;
        ColorType color = q.front().second;
        q.pop();

        ans[curr] = (ans[curr] >= 0 ? min(ans[curr], step) : step);

        auto &edges = (color == ColorType::red ? edges_b : edges_r);
        auto &seen = (color == ColorType::red ? seen_b : seen_r);

        for (int nxt : edges[curr]) {
          if (seen.count(nxt) != 0) continue;
          seen.insert(nxt);
          q.emplace(nxt,
                    color == ColorType::red ? ColorType::blue : ColorType::red);
        }
      }
      ++step;
    }

    return ans;
  }
};

// TODO: 1263. 推箱子
class Solution_1263 {
 public:
  int minPushBox(vector<vector<char>> &grid) { return 0; }
};

// 2. 两数相加
class Solution_2 {
 public:
  ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    ListNode root, *curr = &root;
    int p = 0;
    while (l1 != nullptr || l2 != nullptr || p != 0) {
      int sum =
          (l1 == nullptr ? 0 : l1->val) + (l2 == nullptr ? 0 : l2->val) + p;

      curr->next = new ListNode(sum % 10, nullptr);
      p = sum / 10;
      curr = curr->next;
      if (l1 != nullptr) l1 = l1->next;
      if (l2 != nullptr) l2 = l2->next;
    }
    return root.next;
  }
};

// 445. 两数相加 II
class Solution_445 {
 private:
  ListNode *reverseList(ListNode *root) {
    ListNode *curr = root, *pre = nullptr, *tmp = nullptr;

    while (curr != nullptr) {
      tmp = curr->next;

      curr->next = pre;

      pre = curr;
      curr = tmp;
    }
    return pre;
  }

 public:
  ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
    l1 = reverseList(l1);
    l2 = reverseList(l2);

    ListNode root, *curr = &root;
    int p = 0;
    while (l1 != nullptr || l2 != nullptr || p != 0) {
      int sum =
          (l1 == nullptr ? 0 : l1->val) + (l2 == nullptr ? 0 : l2->val) + p;

      curr->next = new ListNode(sum % 10, nullptr);
      p = sum / 10;
      curr = curr->next;
      if (l1 != nullptr) l1 = l1->next;
      if (l2 != nullptr) l2 = l2->next;
    }
    return reverseList(root.next);
  }
};

// ----------------------- 9.30 -----------------------

// 24. 两两交换链表中的节点
class Solution_24 {
 public:
  ListNode *swapPairs(ListNode *head) {
    if (head == nullptr) return nullptr;
    if (head->next == nullptr) return head;

    ListNode root, *curr = &root;
    root.next = head;

    while (curr != nullptr && curr->next != nullptr &&
           curr->next->next != nullptr) {
      auto nxt = curr->next;
      auto nnxt = nxt->next;

      nxt->next = nnxt->next;
      nnxt->next = nxt;

      curr->next = nnxt;
      curr = nxt;
    }

    return root.next;
  }
};

// 206. 反转链表
class Solution_206 {
 public:
  ListNode *reverseList(ListNode *head) {
    if (head == nullptr) return nullptr;

    ListNode *curr = head, *pre = nullptr, *tmp = nullptr;
    while (curr != nullptr) {
      tmp = curr->next;

      curr->next = pre;

      pre = curr;
      curr = tmp;
    }
    return pre;
  }
};

// 141. 环形链表
class Solution_141 {
 public:
  bool hasCycle(ListNode *head) {
    if (head == nullptr) return false;
    ListNode *fast = head, *slow = head;
    while (fast != nullptr && fast->next != nullptr && slow != nullptr) {
      fast = fast->next->next;
      slow = slow->next;
      if (fast == slow) return true;
    }
    return false;
  }
};

// 142. 环形链表 II
class Solution_142 {
 public:
  ListNode *detectCycle(ListNode *head) {
    if (head == nullptr) return nullptr;
    ListNode *fast = head, *slow = head;
    while (fast != nullptr && fast->next != nullptr && slow != nullptr) {
      fast = fast->next->next;
      slow = slow->next;
      if (fast == slow) {
        ListNode *tmp = head;
        while (tmp != slow) {
          tmp = tmp->next;
          slow = slow->next;
        }

        return tmp;
      }
    }
    return nullptr;
  }
};

// 23. 合并K个升序链表
class Solution_23 {
 private:
  ListNode *mergeLists(ListNode *l1, ListNode *l2) {
    ListNode root, *curr = &root;
    while (l1 != nullptr && l2 != nullptr) {
      if (l1->val < l2->val) {
        curr->next = l1;
        l1 = l1->next;
      } else {
        curr->next = l2;
        l2 = l2->next;
      }
      curr = curr->next;
    }

    if (l1 != nullptr) curr->next = l1;
    if (l2 != nullptr) curr->next = l2;
    while (curr->next != nullptr) curr = curr->next;

    return root.next;
  }

 public:
  ListNode *mergeKLists(vector<ListNode *> &lists) {
    if (lists.empty() == true) return nullptr;
    while (lists.size() > 1) {
      vector<ListNode *> tmp;
      for (int i = 0; i + 1 < lists.size(); i += 2)
        tmp.push_back(mergeLists(lists[i], lists[i + 1]));

      if (lists.size() % 2 == 1) tmp.push_back(lists.back());

      lists = tmp;
    }

    return lists.front();
  }
};

// 21. 合并两个有序链表
class Solution_21 {
 public:
  ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
    ListNode root, *curr = &root;
    while (l1 != nullptr && l2 != nullptr) {
      if (l1->val < l2->val) {
        curr->next = l1;
        l1 = l1->next;
      } else {
        curr->next = l2;
        l2 = l2->next;
      }
      curr = curr->next;
    }

    if (l1 != nullptr) curr->next = l1;
    if (l2 != nullptr) curr->next = l2;
    while (curr->next != nullptr) curr = curr->next;

    return root.next;
  }
};

// 147. 对链表进行插入排序
class Solution_147 {
 public:
  ListNode *insertionSortList(ListNode *head) {
    if (head == nullptr) return nullptr;
    ListNode *curr = head->next;
    head->next = nullptr;

    while (curr != nullptr) {
      ListNode *pre = nullptr, *tmp = head;
      while (tmp != nullptr && tmp->val < curr->val) {
        pre = tmp;
        tmp = tmp->next;
      }
      if (pre == nullptr) {
        tmp = curr->next;
        curr->next = head;
        head = curr;
        curr = tmp;
      } else if (tmp == nullptr) {
        pre->next = curr;
        curr = curr->next;
        pre->next->next = nullptr;
      } else {
        pre->next = curr;
        pre = curr->next;
        curr->next = tmp;

        curr = pre;
      }
    }
    return head;
  }
};

// 148. 排序链表
class Solution_148 {
 private:
  ListNode *mergeLists(ListNode *l1, ListNode *l2) {
    ListNode root, *curr = &root;
    while (l1 != nullptr && l2 != nullptr) {
      if (l1->val < l2->val) {
        curr->next = l1;
        l1 = l1->next;
      } else {
        curr->next = l2;
        l2 = l2->next;
      }
      curr = curr->next;
    }

    if (l1 != nullptr) curr->next = l1;
    if (l2 != nullptr) curr->next = l2;
    while (curr->next != nullptr) curr = curr->next;

    return root.next;
  }

  ListNode *mergeKLists(vector<ListNode *> &lists) {
    if (lists.empty() == true) return nullptr;
    while (lists.size() > 1) {
      vector<ListNode *> tmp;
      for (int i = 0; i + 1 < lists.size(); i += 2)
        tmp.push_back(mergeLists(lists[i], lists[i + 1]));

      if (lists.size() % 2 == 1) tmp.push_back(lists.back());

      lists = tmp;
    }

    return lists.front();
  }

 public:
  ListNode *sortList(ListNode *head) {
    vector<ListNode *> lists;
    ListNode *curr = head;

    while (curr != nullptr) {
      lists.push_back(curr);
      while (curr != nullptr && curr->next != nullptr &&
             curr->val < curr->next->val)
        curr = curr->next;
      if (curr != nullptr) {
        ListNode *tmp = curr->next;
        curr->next = nullptr;
        curr = tmp;
      }
    }
    return mergeKLists(lists);
  }
};

// 707. 设计链表
class MyLinkedList {
 private:
  class Node {
   public:
    Node() : val(0), next(nullptr) {}
    Node(int v) : val(v), next(nullptr) {}
    Node(int v, Node *n) : val(v), next(n) {}

    int val;
    Node *next;
  };

  Node *head, *tail;
  Node dummy;
  int max_index;

  Node *getNode(int index) {
    dummy.next = head;
    Node *curr = &dummy;
    while (index >= 0) {
      curr = curr->next;
      --index;
    }
    return curr;
  }

 public:
  MyLinkedList() : head(nullptr), tail(nullptr), max_index(-1), dummy(0) {}

  ~MyLinkedList() {
    Node *node = head;
    while (node != nullptr) {
      Node *curr = node;
      node = node->next;
      delete curr;
    }
    head = nullptr;
    tail = nullptr;
  }

  int get(int index) {
    if (index < 0 || index > max_index) return -1;
    return getNode(index)->val;
  }

  void addAtHead(int val) {
    head = new Node(val, head);
    if (max_index++ == -1) tail = head;
  }

  void addAtTail(int val) {
    Node *node = new MyLinkedList::Node(val);
    if (max_index++ == -1) {
      head = tail = node;
    } else {
      tail->next = node;
      tail = tail->next;
    }
  }

  void addAtIndex(int index, int val) {
    if (index < 0 || index > (max_index + 1)) return;
    if (index == 0) return addAtHead(val);
    if (index == (max_index + 1)) return addAtTail(val);

    Node *prev = getNode(index - 1);
    prev->next = new Node(val, prev->next);
    ++max_index;
  }

  void deleteAtIndex(int index) {
    if (index < 0 || index > max_index) return;
    Node *prev = getNode(index - 1), *delete_node = prev->next;
    prev->next = delete_node->next;

    if (index == 0) head = prev->next;
    if (index == max_index) tail = prev;
    delete delete_node;
    --max_index;
  }
};

// ----------------------- 10.1 -----------------------
// 35. 搜索插入位置
class Solution_35 {
 public:
  int searchInsert(vector<int> &nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
      int m = l + (r - l) / 2;
      if (nums[m] == target)
        return m;
      else if (nums[m] > target)
        r = m;
      else
        l = m + 1;
    }

    return l;
  }
};

// TODO: 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution_34 {
 private:
  int binarySearch(vector<int> &nums, int target, bool lower) {
    int l = 0, r = nums.size();
    while (l <= r) {
      int m = l + (r - l) / 2;
    }
    return l;
  }

 public:
  vector<int> searchRange(vector<int> &nums, int target) { return {}; }
};

// 704. 二分查找
class Solution_704 {
 private:
  int searchInsert(vector<int> &nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
      int m = l + (r - l) / 2;
      if (nums[m] == target)
        return m;
      else if (nums[m] > target)
        r = m;
      else
        l = m + 1;
    }

    return l;
  }

 public:
  int search(vector<int> &nums, int target) {
    int index = searchInsert(nums, target);
    if (index >= nums.size() || nums[index] != target) return -1;
    return index;
  }
};

// 981. 基于时间的键值存储
class TimeMap {
 private:
  unordered_map<string, map<int, string>> m;

 public:
  TimeMap() {}

  void set(string key, string value, int timestamp) {
    m[key].emplace(timestamp, std::move(value));
  }

  string get(string key, int timestamp) {
    auto item = m.find(key);
    if (item == m.end()) return "";

    auto it = item->second.upper_bound(timestamp);
    if (it == item->second.begin()) return "";
    return (--it)->second;
  }
};

// TODO: 33. 搜索旋转排序数组
class Solution_33 {
 public:
  int search(vector<int> &nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
      int m = l + (r - l) / 2;
      if (nums[m] == target) return m;

      if (nums[m] >= nums.front()) {
        if (nums.front() <= target && target < nums[m]) {
          r = m;
        } else {
          l = m + 1;
        }
      } else {
        if (nums[m] < target && target <= nums.back()) {
          l = m + 1;
        } else {
          r = m;
        }
      }
    }
    return -1;
  }
};

// ----------------------- 10.11 -----------------------

// TODO: 81. 搜索旋转排序数组 II
// 含有重复元素导致无法判断大小端
class Solution_81 {
 public:
  bool search(vector<int> &nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
      int m = l + (r - l) / 2;
      if (nums[m] == target) return true;

      // 去掉干扰项
      if (nums[m] == nums[l]) {
        ++l;
        continue;
      }

      if (nums[m] > nums.front()) {
        if (nums.front() <= target && target < nums[m]) {
          r = m;
        } else {
          l = m + 1;
        }
      } else {
        if (nums[m] < target && target <= nums.back()) {
          l = m + 1;
        } else {
          r = m;
        }
      }
    }
    return false;
  }
};

// 153. 寻找旋转排序数组中的最小值
class Solution_153 {
 public:
  int findMin(vector<int> &nums) {
    if (nums.size() == 1) return nums.front();
    if (nums.front() < nums.back()) return nums.front();

    int l = 0, r = nums.size();
    while (l < r) {
      int m = l + (r - l) / 2;
      if (l > 0 && nums[l - 1] > nums[l]) return nums[l];

      if (nums.front() < nums[m]) {
        l = m + 1;
      } else {
        ++l;
      }
    }

    return -1;
  }
};

// 154. 寻找旋转排序数组中的最小值 II
class Solution_154 {
 private:
  int findMin(vector<int> &nums, int l, int r) {
    if ((l + 1) >= r) return min(nums[l], nums[r]);

    if (nums[l] < nums[r]) return nums[l];

    int m = l + (r - l) / 2;

    return min(findMin(nums, m, r), findMin(nums, l, m - 1));
  }

 public:
  int findMin(vector<int> &nums) { return findMin(nums, 0, nums.size() - 1); }
};

// 162. 寻找峰值
class Solution_162 {
 public:
  int findPeakElement(vector<int> &nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
      int m = l + (r - l) / 2;

      if (nums[m] > nums[m + 1])
        r = m;
      else
        l = m + 1;
    }

    return l;
  }
};

// 852. 山脉数组的峰顶索引
class Solution_852 {
 public:
  int peakIndexInMountainArray(vector<int> &arr) {
    int l = 0, r = arr.size();

    while (l < r) {
      int m = l + (r - l) / 2;
      if (arr[m] > arr[m + 1])
        r = m;
      else
        l = m + 1;
    }
    return l;
  }
};

// 69. Sqrt(x)
class Solution_69 {
 public:
  int mySqrt(int x) {
    long long l = 1, r = static_cast<long long>(x) + 1;
    while (l < r) {
      long long m = l + (r - l) / 2;

      if (m * m > x)
        r = m;
      else
        l = m + 1;
    }
    return l - 1;
  }
};

// 74. 搜索二维矩阵
class Solution_74 {
 public:
  bool searchMatrix(vector<vector<int>> &matrix, int target) {
    if (matrix.empty() == true) return false;
    int cols = matrix.front().size();
    int r = matrix.size() * cols, l = 0;
    while (l < r) {
      int m = l + (r - l) / 2;
      if (matrix[m / cols][m % cols] == target)
        return true;
      else if (matrix[m / cols][m % cols] > target)
        r = m;
      else
        l = m + 1;
    }
    return false;
  }
};

// 875. 爱吃香蕉的珂珂
class Solution_875 {
 public:
  int minEatingSpeed(vector<int> &piles, int h) {
    int l = 1, r = *max_element(piles.begin(), piles.end()) + 1;
    while (l < r) {
      int m = l + (r - l) / 2, h_tmp = 0;
      for (auto p : piles) h_tmp += (p + m - 1) / m;
      if (h_tmp > h)
        l = m + 1;
      else
        r = m;
    }
    return l;
  }
};

// 11. 盛最多水的容器
class Solution_11 {
 public:
  int maxArea(vector<int> &height) {
    int l = 0, r = height.size() - 1, ans = 0;
    while (l < r) {
      ans = max(ans, (r - l) * min(height[l], height[r]));
      if (height[l] > height[r])
        --r;
      else
        ++l;
    }

    return ans;
  }
};

// 42. 接雨水
class Solution_42 {
 public:
  int trap(vector<int> &height) {
    int n = height.size();
    if (n == 0) return 0;
    stack<int> s;

    int ans = 0;
    for (int i = 0; i < n; ++i) {
      while (s.empty() == false && height[i] > height[s.top()]) {
        int top = s.top();
        s.pop();
        if (s.empty() == true) break;
        int left = s.top();
        int w = i - left - 1, h = min(height[i], height[left]) - height[top];
        ans += w * h;
      }
      s.push(i);
    }
    return ans;
  }
};

// 125. 验证回文串
class Solution_125 {
 public:
  bool isPalindrome(string s) {
    int l = 0, r = s.size() - 1;
    while (l < r) {
      if (isalnum(s[l]) == false) {
        ++l;
        continue;
      }
      if (isalnum(s[r]) == false) {
        --r;
        continue;
      }

      if (tolower(s[l]) != tolower(s[r])) return false;
      ++l;
      --r;
    }
    return true;
  }
};

// ----------------------- 10.13 -----------------------

// 917. 仅仅反转字母
class Solution_917 {
 public:
  string reverseOnlyLetters(string s) {
    int l = 0, r = s.size() - 1;

    while (l < r) {
      if (isalpha(s[l]) == false) {
        ++l;
        continue;
      }
      if (isalpha(s[r]) == false) {
        --r;
        continue;
      }

      swap(s[l++], s[r--]);
    }

    return s;
  }
};

// 925. 长按键入
class Solution_925 {
 public:
  bool isLongPressedName(string name, string typed) {
    int i = 0, j = 0, n1 = name.size(), n2 = typed.size();
    while (i < n1 && j < n2) {
      if (name[i] == typed[j]) {
        ++i;
        ++j;
      } else if (i > 0 && name[i - 1] == typed[j])
        ++j;
      else
        return false;
    }
    while (j < n2 && typed[j] == typed[j - 1]) ++j;
    return i == n1 && j == n2;
  }
};

// TODO: 986. 区间列表的交集
class Solution_986 {
 public:
  vector<vector<int>> intervalIntersection(vector<vector<int>> &firstList,
                                           vector<vector<int>> &secondList) {
    if (firstList.empty() == true || secondList.empty() == true) return {};
    int i = 0, j = 0, n1 = firstList.size(), n2 = secondList.size();

    vector<vector<int>> ans;

    while (i < n1 && j < n2) {
      int s = max(firstList[i].front(), secondList[j].front()),
          e = min(firstList[i].back(), secondList[j].back());
      if (s <= e) ans.push_back({s, e});
      if (firstList[i].back() < secondList[j].back())
        ++i;
      else
        ++j;
    }

    return ans;
  }
};

// 167. 两数之和 II - 输入有序数组
class Solution_167 {
 public:
  vector<int> twoSum(vector<int> &numbers, int target) {
    int n = numbers.size(), l = 0, r = n - 1;
    while (l < r) {
      int sum = numbers[l] + numbers[r];
      if (sum == target) return {l + 1, r + 1};
      if (sum < target)
        ++l;
      else
        --r;
    }
    return {};
  }
};

// 15. 三数之和
class Solution_15 {
 public:
  vector<vector<int>> threeSum(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> ans;
    for (int i = 0; i < n - 2; ++i) {
      if (nums[i] > 0) break;
      if (i > 0 && nums[i] == nums[i - 1]) continue;
      int l = i + 1, r = n - 1;
      while (l < r) {
        if ((nums[i] + nums[l] + nums[r]) == 0) {
          ans.push_back({nums[i], nums[l++], nums[r--]});
          while (l < r && nums[l] == nums[l - 1]) ++l;
          while (l < r && nums[r] == nums[r + 1]) --r;
        } else if (nums[i] + nums[l] + nums[r] < 0)
          ++l;
        else
          --r;
      }
    }

    return ans;
  }
};

// 16. 最接近的三数之和
class Solution_16 {
 public:
  int threeSumClosest(vector<int> &nums, int target) {
    sort(nums.begin(), nums.end());
    int ans = target, change = INT_MAX;
    const int n = nums.size();
    for (int i = 0; i < n - 2; ++i) {
      if (i > 0 && nums[i] == nums[i - 1]) continue;
      int l = i + 1, r = n - 1;
      while (l < r) {
        int sum = nums[i] + nums[l] + nums[r];
        if (sum == target) return target;

        int diff = abs(sum - target);
        if (diff < change) {
          change = diff;
          ans = sum;
        }

        if (sum > target)
          --r;
        else
          ++l;
      }
    }

    return ans;
  }
};

// 977. 有序数组的平方
class Solution_977 {
 public:
  vector<int> sortedSquares(vector<int> &nums) {
    int l = 0, r = nums.size() - 1;
    vector<int> ans;
    while (l <= r) {
      if (abs(nums[l]) > abs(nums[r])) {
        ans.push_back(nums[l] * nums[l]);
        ++l;
      } else {
        ans.push_back(nums[r] * nums[r]);
        --r;
      }
    }
    reverse(ans.begin(), ans.end());
    return ans;
  }
};

// ----------------------- 10.16 -----------------------

// 17. 电话号码的字母组合
class Solution_17 {
 private:
  map<char, vector<string>> store{
      {'2', {"a", "b", "c"}}, {'3', {"d", "e", "f"}},
      {'4', {"g", "h", "i"}}, {'5', {"j", "k", "l"}},
      {'6', {"m", "n", "o"}}, {'7', {"p", "q", "r", "s"}},
      {'8', {"t", "u", "v"}}, {'9', {"w", "x", "y", "z"}}};
  vector<string> ans;

  void back_trace(string &&curr, int index, string digits) {
    if (curr.size() == digits.size()) {
      ans.push_back(curr);
      return;
    }

    vector<string> letters = store[digits[index]];
    for (auto letter : letters) {
      curr += letter;
      back_trace(std::move(curr), index + 1, digits);
      curr.pop_back();
    }
  }

 public:
  vector<string> letterCombinations(string digits) {
    if (digits.empty() == true) return {};
    back_trace("", 0, digits);
    return ans;
  }
};

// 39. 组合总和
class Solution_39 {
 private:
  vector<vector<int>> ans;

  void back_trace(vector<int> &curr, int target, int index,
                  const vector<int> &candidates) {
    if (index >= candidates.size()) return;
    if (target == 0) {
      ans.push_back(curr);
      return;
    }

    if (target - candidates[index] >= 0) {
      curr.push_back(candidates[index]);
      back_trace(curr, target - candidates[index], index, candidates);
      curr.pop_back();
    }
    back_trace(curr, target, index + 1, candidates);
  }

 public:
  vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
    vector<int> curr;
    back_trace(curr, target, 0, candidates);
    return ans;
  }
};

// 40. 组合总和 II
class Solution_40 {
 private:
  set<vector<int>> ans;

  void back_trace(const vector<int> &candidates, vector<int> &curr, int index,
                  int target) {
    if (target == 0) {
      ans.insert(curr);
      return;
    }

    for (int i = index; i < candidates.size(); ++i) {
      if (candidates[i] > target) break;
      curr.push_back(candidates[i]);
      back_trace(candidates, curr, i + 1, target - candidates[i]);
      curr.pop_back();
    }
  }

 public:
  vector<vector<int>> combinationSum2(vector<int> &candidates, int target) {
    sort(candidates.begin(), candidates.end());
    vector<int> curr;
    back_trace(candidates, curr, 0, target);
    return vector<vector<int>>{ans.begin(), ans.end()};
  }
};

// 77. 组合
class Solution_77 {
 public:
  vector<vector<int>> combine(int n, int k) {
    vector<int> curr;
    back_trace(n, k, curr);
    return ans;
  }

 private:
  vector<vector<int>> ans;

  void back_trace(int n, int k, vector<int> &curr) {
    if (k == 0) {
      ans.push_back(curr);
      return;
    }

    for (int i = n; i >= 1; --i) {
      curr.push_back(i);
      back_trace(i - 1, k - 1, curr);
      curr.pop_back();
    }
  }
};

// 78. 子集
class Solution_78 {
 public:
  vector<vector<int>> subsets(vector<int> &nums) {
    for (int i = 0; i <= nums.size(); ++i) {
      vector<int> curr;
      back_trace(nums, curr, i, 0);
    }
    return ans;
  }

 private:
  vector<vector<int>> ans;

  void back_trace(const vector<int> &nums, vector<int> &curr, int n,
                  int index) {
    if (n == 0) {
      ans.push_back(curr);
      return;
    }

    for (int i = index; i < nums.size(); ++i) {
      curr.push_back(nums[i]);
      back_trace(nums, curr, n - 1, i + 1);
      curr.pop_back();
    }
  }
};

// 90. 子集 II
class Solution_90 {
 public:
  vector<vector<int>> subsetsWithDup(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    for (int i = 0; i <= nums.size(); ++i) {
      vector<int> curr;
      back_trace(nums, curr, i, 0);
    }

    return vector<vector<int>>{ans.begin(), ans.end()};
  }

 private:
  set<vector<int>> ans;

  void back_trace(const vector<int> &nums, vector<int> &curr, int n,
                  int index) {
    if (n == 0) {
      ans.insert(curr);
      return;
    }

    for (int i = index; i < nums.size(); ++i) {
      curr.push_back(nums[i]);
      back_trace(nums, curr, n - 1, i + 1);
      curr.pop_back();
    }
  }
};

// 216. 组合总和 III
class Solution_216 {
 public:
  vector<vector<int>> combinationSum3(int k, int n) {}

 private:
  vector<vector<int>> ans;

  void back_trace(vector<int> &curr, int k, int target, int index) {
    if (k == 0 && target == 0) {
      ans.push_back(curr);
      return;
    }
  }
};

int main() {
  std::cout << "It's ok!" << std::endl;
  return 0;
}
