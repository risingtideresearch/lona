import { describe, expect, test } from "vitest";
import { stackTraversal } from "./traversal"; // Adjust the import path as needed

// Define a simple tree node structure for testing
interface TestNode {
  value: string;
  children: TestNode[];
}

describe("stackTraversal", () => {
  // Helper function to create a simple node
  const createNode = (value: string, children: TestNode[] = []): TestNode => ({
    value,
    children,
  });

  // Helper function to get children from a node
  const getChildren = (node: TestNode): TestNode[] => node.children;

  describe("Basic functionality", () => {
    test("should traverse a single node tree", () => {
      // Setup
      const root = createNode("A");
      const preOrder: string[] = [];
      const postOrder: string[] = [];

      // Execute
      stackTraversal(root, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      // Assert
      expect(preOrder).toEqual(["A"]);
      expect(postOrder).toEqual(["A"]);
    });

    test("should traverse a simple tree", () => {
      /*
          A
         / \
        B   C
      */
      const nodeC = createNode("C");
      const nodeB = createNode("B");
      const nodeA = createNode("A", [nodeB, nodeC]);

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      expect(preOrder).toEqual(["A", "B", "C"]);
      expect(postOrder).toEqual(["B", "C", "A"]);
    });

    test("should traverse a deeper tree", () => {
      /*
            A
           / \
          B   C
         /   / \
        D   E   F
            |
            G
      */
      const nodeG = createNode("G");
      const nodeE = createNode("E", [nodeG]);
      const nodeF = createNode("F");
      const nodeD = createNode("D");
      const nodeC = createNode("C", [nodeE, nodeF]);
      const nodeB = createNode("B", [nodeD]);
      const nodeA = createNode("A", [nodeB, nodeC]);

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      expect(preOrder).toEqual(["A", "B", "D", "C", "E", "G", "F"]);
      expect(postOrder).toEqual(["D", "B", "G", "E", "F", "C", "A"]);
    });
  });

  describe("Edge cases", () => {
    test("should handle a tree with duplicate node references", () => {
      // Create a tree with duplicate references
      const nodeC = createNode("C");
      const nodeB = createNode("B", [nodeC, nodeC]); // Same node C appears twice
      const nodeA = createNode("A", [nodeB]);

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      // Node C should only appear once in the traversal
      expect(preOrder).toEqual(["A", "B", "C"]);
      expect(postOrder).toEqual(["C", "B", "A"]);
    });

    test("should handle a cyclic graph", () => {
      // Create a graph with a cycle: A -> B -> C -> A
      const nodeA: TestNode = createNode("A");
      const nodeB = createNode("B");
      const nodeC = createNode("C");

      // Create a cycle
      nodeA.children = [nodeB];
      nodeB.children = [nodeC];
      nodeC.children = [nodeA];

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      // Each node should only appear once
      expect(preOrder).toEqual(["A", "B", "C"]);
      expect(postOrder).toEqual(["C", "B", "A"]);
    });

    test("should handle a diamond-shaped graph", () => {
      /*
            A
           / \
          B   C
           \ /
            D
      */
      const nodeD = createNode("D");
      const nodeC = createNode("C", [nodeD]);
      const nodeB = createNode("B", [nodeD]);
      const nodeA = createNode("A", [nodeB, nodeC]);

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      // D should only appear once
      expect(preOrder).toEqual(["A", "B", "D", "C"]);
      expect(postOrder).toEqual(["D", "B", "C", "A"]);
    });
  });

  describe("Callback options", () => {
    test("should work with only preCallBack", () => {
      const nodeC = createNode("C");
      const nodeB = createNode("B");
      const nodeA = createNode("A", [nodeB, nodeC]);

      const preOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
      });

      expect(preOrder).toEqual(["A", "B", "C"]);
    });

    test("should work with only postCallBack", () => {
      const nodeC = createNode("C");
      const nodeB = createNode("B");
      const nodeA = createNode("A", [nodeB, nodeC]);

      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        postCallBack: (node) => postOrder.push(node.value),
      });

      expect(postOrder).toEqual(["B", "C", "A"]);
    });

    test("should work with no callbacks", () => {
      const nodeC = createNode("C");
      const nodeB = createNode("B");
      const nodeA = createNode("A", [nodeB, nodeC]);

      // This should not throw, even with no callbacks
      expect(() => {
        stackTraversal(nodeA, getChildren);
      }).not.toThrow();
    });
  });

  describe("Order verification", () => {
    test("should visit children in the specified order", () => {
      /*
            A
           /|\
          B C D
         /|
        E F
      */
      const nodeE = createNode("E");
      const nodeF = createNode("F");
      const nodeB = createNode("B", [nodeE, nodeF]);
      const nodeC = createNode("C");
      const nodeD = createNode("D");
      const nodeA = createNode("A", [nodeB, nodeC, nodeD]);

      const visitOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => visitOrder.push(`pre-${node.value}`),
        postCallBack: (node) => visitOrder.push(`post-${node.value}`),
      });

      // Verify the exact order of pre and post callbacks
      expect(visitOrder).toEqual([
        "pre-A",
        "pre-B",
        "pre-E",
        "post-E",
        "pre-F",
        "post-F",
        "post-B",
        "pre-C",
        "post-C",
        "pre-D",
        "post-D",
        "post-A",
      ]);
    });

    test("should correctly process large branching factors", () => {
      // Create a wide tree with many children
      const children: TestNode[] = [];
      for (let i = 0; i < 100; i++) {
        children.push(createNode(`Child${i}`));
      }
      const root = createNode("Root", children);

      const visitedPre: string[] = [];
      const visitedPost: string[] = [];

      stackTraversal(root, getChildren, {
        preCallBack: (node) => visitedPre.push(node.value),
        postCallBack: (node) => visitedPost.push(node.value),
      });

      // Should have 101 nodes in total (1 root + 100 children)
      expect(visitedPre.length).toBe(101);
      expect(visitedPost.length).toBe(101);

      // Root should be first in pre-order
      expect(visitedPre[0]).toBe("Root");

      // Root should be last in post-order
      expect(visitedPost[visitedPost.length - 1]).toBe("Root");

      // All children should be visited
      for (let i = 0; i < 100; i++) {
        expect(visitedPre).toContain(`Child${i}`);
        expect(visitedPost).toContain(`Child${i}`);
      }
    });
  });

  describe("Advanced scenarios", () => {
    test("should handle deep trees without stack overflow", () => {
      // Create a deep tree
      let current = createNode("root");
      const root = current;

      // Create a deep chain of 10,000 nodes
      for (let i = 0; i < 10000; i++) {
        const next = createNode(`node${i}`);
        current.children = [next];
        current = next;
      }

      let nodeCount = 0;

      // This should complete without stack overflow
      stackTraversal(root, getChildren, {
        preCallBack: () => nodeCount++,
      });

      expect(nodeCount).toBe(10001); // root + 10000 nodes
    });

    test("should handle trees with unequal depths", () => {
      /*
              A
             / \
            B   C
           /     \
          D       E
         /         \
        F           G
                     \
                      H
      */
      const nodeH = createNode("H");
      const nodeG = createNode("G", [nodeH]);
      const nodeE = createNode("E", [nodeG]);
      const nodeF = createNode("F");
      const nodeD = createNode("D", [nodeF]);
      const nodeC = createNode("C", [nodeE]);
      const nodeB = createNode("B", [nodeD]);
      const nodeA = createNode("A", [nodeB, nodeC]);

      const preOrder: string[] = [];
      const postOrder: string[] = [];

      stackTraversal(nodeA, getChildren, {
        preCallBack: (node) => preOrder.push(node.value),
        postCallBack: (node) => postOrder.push(node.value),
      });

      expect(preOrder).toEqual(["A", "B", "D", "F", "C", "E", "G", "H"]);
      expect(postOrder).toEqual(["F", "D", "B", "H", "G", "E", "C", "A"]);
    });
  });
});
