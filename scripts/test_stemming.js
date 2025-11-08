import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

const queries = new SharedArray('queries', function () {
  return open('../data/queries.txt').split('\n');
});

export default function () {
  const query = queries[Math.floor(Math.random() * queries.length)];
  const data = {
    "input-word": query
  };

  const res = http.post(`https://am.unpak.ac.id/stemming_word`, data);
  check(res, {
    'HTTP request ok': (r) => r.status === 200,
  });

  sleep(1);
}
